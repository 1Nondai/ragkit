"""图节点：每个节点只做一件事，输入 state，输出**增量** state。"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragkit.config import Settings, get_settings
from ragkit.graph.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    GRADE_SYSTEM,
    GRADE_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    VERIFY_SYSTEM,
    VERIFY_USER,
    build_messages,
    format_history,
)
from ragkit.graph.router import HybridRouter
from ragkit.llm import ChatModel, ExtractiveLLM, supports_vision
from ragkit.logging_setup import get_logger
from ragkit.multimodal.image_store import image_data_url
from ragkit.retrieval.service import RetrievalService

log = get_logger("graph.nodes")

@dataclass
class Deps:
    """节点依赖集合（一次构造，多处复用）。"""

    service: RetrievalService
    chat: ChatModel
    router: HybridRouter | None = None
    settings: Settings = field(default_factory=get_settings)
    history: HistoryStore | None = None
    use_llm_judge: bool = True
    wiki_router: Any = None
    wiki_manager: Any = None
    wiki_chunk_index: dict = field(default_factory=dict)
    image_root: Path | None = None

    @property
    def offline(self) -> bool:
        """离线模式（无真实生成能力）时，跳过所有模型判定，走确定性逻辑。"""
        return isinstance(self.chat, ExtractiveLLM)

class HistoryStore:

    def __init__(self, max_turns: int = 10) -> None:
        self._data: dict[str, list[str]] = {}
        self.max_turns = max_turns

    def get(self, session_id: str) -> list[str]:
        """取某个会话的历史（问题/答案交替存放）。"""
        return list(self._data.get(session_id, []))

    def append(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮问答，并按 max_turns 裁剪最旧的记录。"""
        buf = self._data.setdefault(session_id, [])
        buf.extend([question, answer])
        if len(buf) > self.max_turns * 2:
            del buf[: len(buf) - self.max_turns * 2]

@contextmanager
def _timed(state: dict[str, Any], name: str):
    """记录节点耗时并写入 state['timings']（毫秒）。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        state.setdefault("timings", {})[name] = round(elapsed, 2)
        state.setdefault("trace", []).append(name)

def node_route(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """路由：判断走知识库 / 闲聊 / 追问。"""
    with _timed(state, "route"):
        if not deps.settings.graph.enable_route or deps.router is None:
            return {"route": "kb", "trace": state.get("trace", [])}

        question = state.get("question", "")
        history = deps.history.get(state.get("session_id", "default")) if deps.history else []
        decision = deps.router.decide(question, history)

        update: dict[str, Any] = {
            "route": decision.route,
            "trace": state.get("trace", []),
            "timings": state.get("timings", {}),
        }

        if decision.kb_name:
            update["route_kb"] = decision.kb_name
        log.info("路由结果: %s（%s）", decision.route, decision.reason)
        return update

def node_wiki(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """Wiki 直答：命中已编译知识页就直接结束，不命中把控制权交回检索链路。"""
    with _timed(state, "wiki"):
        updates: dict[str, Any] = {"trace": state.get("trace", []),
                                   "timings": state.get("timings", {}),
                                   "wiki_hit": False}
        router = deps.wiki_router
        if router is None or not getattr(deps.settings.wiki, "enabled", False):
            updates["wiki_reason"] = "wiki 层未启用"
            return updates

        question = state.get("question", "")
        try:
            decision = router.decide(question)
        except Exception as exc:  # noqa: BLE001 - 增强层出错绝不能拖垮主链路
            log.warning("wiki 路由异常，退回检索: %s", exc)
            updates["wiki_reason"] = f"路由异常: {exc}"
            return updates

        if deps.wiki_manager is not None:
            try:
                deps.wiki_manager.record_query(decision.hit, decision.slug)
            except Exception as exc:  # noqa: BLE001 - 统计失败不影响回答
                log.debug("wiki 命中统计写入失败: %s", exc)

        updates["wiki_score"] = round(float(decision.score), 4)
        updates["wiki_reason"] = decision.reason

        if not decision.hit or decision.page is None:
            updates["wiki_gate"] = decision.gate
            return updates

        page = decision.page
        citations = [
            {"source": src.split("#")[0], "title": page.title,
             "chunk_id": src.split("#")[-1], "score": round(decision.score, 4)}
            for src in page.sources[:5]
        ] or [{"source": "wiki", "title": page.title, "chunk_id": page.slug,
               "score": round(decision.score, 4)}]
        updates.update(
            {
                "wiki_hit": True,
                "wiki_slug": decision.slug,
                "answer": router.compose_answer(page),
                "citations": citations,
                "grounded": True,
                "confidence": round(decision.score, 3),
            }
        )
        return updates

def node_rewrite(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """查询改写/扩展：把一句话变成 N 个互补的检索查询。"""
    with _timed(state, "rewrite"):
        question = state.get("question", "")
        count = int(state.get("rewrite_count", 0))
        updates: dict[str, Any] = {"trace": state.get("trace", []),
                                   "timings": state.get("timings", {})}

        history = deps.history.get(state.get("session_id", "default")) if deps.history else []
        if deps.offline or count > 0:
            candidates = _rule_expand(question, count)
        else:
            try:
                raw = deps.chat.invoke(
                    build_messages(
                        REWRITE_SYSTEM.format(n=3),
                        REWRITE_USER.format(history=format_history(history),
                                            question=question, n=3),
                    )
                )
                candidates = [line.strip("- *").strip()
                              for line in (raw or "").splitlines() if line.strip()]
            except Exception as exc:  # noqa: BLE001
                log.warning("查询改写失败，使用原问题: %s", exc)
                candidates = []

        if not candidates:
            candidates = [question]

        if question not in candidates:
            candidates.insert(0, question)

        updates["rewritten_queries"] = candidates
        updates["effective_query"] = candidates[0]
        updates["rewrite_count"] = count + 1
        return updates

def _rule_expand(question: str, attempt: int) -> list[str]:
    """规则式扩展（无模型时的兜底）：按标点切分 + 抽取名词性片段。"""
    parts = [p.strip() for p in re.split(r"[，,。；;？?]", question) if len(p.strip()) > 3]
    words = re.findall(r"[A-Za-z0-9\-\.]+", question)
    candidates = [question]
    if parts:
        candidates.append(parts[0])
    if words:
        candidates.append(" ".join(words[:6]))
    if attempt > 0 and len(candidates) > 1:
        candidates = candidates[1:] + candidates[:1]
    return candidates[:3]

def node_retrieve(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """多查询检索 + 结果合并去重。"""
    with _timed(state, "retrieve"):
        queries = state.get("rewritten_queries") or [state.get("question", "")]
        seen: dict[str, dict[str, Any]] = {}
        hits: list[dict[str, Any]] = []
        elapsed = 0.0

        for q in queries:
            ctx = deps.service.search(q, filters=_filters(state))
            elapsed += ctx.elapsed_ms
            for hit in ctx.hits:
                if hit.chunk_id in seen:
                    seen[hit.chunk_id]["score"] = round(
                        seen[hit.chunk_id]["score"] + hit.score, 4
                    )
                    continue
                item = {
                    "chunk_id": hit.chunk_id,
                    "text": hit.text,
                    "score": round(hit.score, 4),
                    "dense_score": round(hit.dense_score, 4),
                    "sparse_score": round(hit.sparse_score, 4),
                    "source": hit.source,
                    "title": hit.title,
                    "metadata": hit.metadata,
                    "modality": hit.modality,
                    "image_path": hit.image_path,
                    "page": hit.page,
                }
                seen[hit.chunk_id] = item
                hits.append(item)

        hits.sort(key=lambda x: x["score"], reverse=True)
        top_k = deps.settings.retrieval.top_k
        hits = hits[:top_k]

        if deps.wiki_manager is not None and getattr(deps.settings.wiki, "enabled", False):
            if not deps.wiki_chunk_index:
                try:
                    deps.wiki_chunk_index.update(deps.wiki_manager.chunk_page_index())
                except Exception as exc:  # noqa: BLE001 - 增强层失败不影响检索
                    log.debug("wiki 反查索引构建失败: %s", exc)
            for item in hits:
                item["wiki_title"] = deps.wiki_chunk_index.get(item["chunk_id"], "")

        context = "\n\n".join(
            f"[{i}] 来源: {h['source']}{(' · ' + h['title']) if h['title'] else ''}\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        log.info("检索完成: %d 条命中（查询数=%d）", len(hits), len(queries))
        return {
            "hits": hits,
            "context": context,
            "retrieval_elapsed_ms": round(elapsed, 2),
            "trace": state.get("trace", []),
            "timings": state.get("timings", {}),
        }

def _filters(state: dict[str, Any]) -> dict[str, Any] | None:
    """只有**显式指定**的 kb_name 才会变成过滤条件（见 node_route 的注释）。"""
    kb = state.get("kb_name")
    return {"kb": kb} if kb else None

def node_grade(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """评估检索结果是否"够用"，决定生成 / 改写重试 / 直接说没有。"""
    with _timed(state, "grade"):
        hits = state.get("hits", [])
        question = state.get("question", "")
        updates: dict[str, Any] = {"trace": state.get("trace", []),
                                   "timings": state.get("timings", {})}

        if not hits:
            updates["grounded"] = False
            updates["confidence"] = 0.0
            return updates

        if deps.offline or not deps.use_llm_judge:
            score = _keyword_grounding(question, hits)
            updates["confidence"] = round(score, 3)
            updates["grounded"] = score >= deps.settings.graph.min_grounded_score
            return updates

        yes = 0
        for hit in hits[:3]:
            try:
                raw = deps.chat.invoke(
                    build_messages(GRADE_SYSTEM,
                                   GRADE_USER.format(document=hit["text"][:1500],
                                                     question=question))
                )
                if "yes" in (raw or "").lower():
                    yes += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("相关性判定失败: %s", exc)
        ratio = yes / min(3, len(hits))
        updates["confidence"] = round(ratio, 3)
        updates["grounded"] = ratio >= deps.settings.graph.min_grounded_score
        return updates

_STOP_CHARS = set("的是了么呢吗我你他它这那有什么如和对与及在为就都也很要会能可以到把被从向")
_EN_STOPWORDS = {"the", "is", "a", "an", "of", "to", "in", "for", "on", "and", "or", "what"}

def _content_terms(text: str) -> set:
    """抽取"内容词"：英文/数字词（去停用词）+ 非虚词构成的中文双字片段。"""
    terms = {
        t for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if t not in _EN_STOPWORDS
    }
    terms |= {
        text[i:i + 2]
        for i in range(len(text or "") - 1)
        if all("\u4e00" <= c <= "\u9fff" for c in text[i:i + 2])
        and not (text[i] in _STOP_CHARS or text[i + 1] in _STOP_CHARS)
    }
    return terms

def _keyword_grounding(question: str, hits: list[dict[str, Any]]) -> float:
    """词面覆盖率（离线判定用）：问句内容词在 top-3 命中文本（合集）中出现的比例。"""
    terms = _content_terms(question)
    if not terms:
        return 0.0
    union = " ".join((h.get("text") or "") for h in hits[:3]).lower()
    return sum(1 for t in terms if t in union) / len(terms)

def _vision_messages(
    system_text: str,
    user_text: str,
    hits: list[dict[str, Any]],
    image_root: Path | None,
) -> list[dict] | None:
    """把命中里的图片块转成 OpenAI 多模态消息。"""
    if not image_root:
        return None
    image_parts: list[dict] = []
    seen: set[str] = set()
    for h in hits[:3]:
        rel = h.get("image_path") or ""
        if not rel or rel in seen:
            continue
        data_url = image_data_url(Path(image_root) / rel)
        if data_url is None:
            continue
        seen.add(rel)
        image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
    if not image_parts:
        return None
    user_content: list[dict] = [{"type": "text", "text": user_text}]
    user_content.append(
        {"type": "text", "text": "附：以下图片来自检索命中的文档页，回答时结合图片内容。"}
    )
    user_content.extend(image_parts)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

def node_generate(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """生成答案（带引用）。"""
    with _timed(state, "generate"):
        question = state.get("question", "")
        context = state.get("context", "")
        updates: dict[str, Any] = {"trace": state.get("trace", []),
                                   "timings": state.get("timings", {})}

        if not context.strip():
            updates["answer"] = (
                "未在知识库中检索到相关内容，无法作答。\n"
                "建议：补充该主题的文档后重新入库，或换一种说法描述问题。"
            )
            updates["grounded"] = False
            return updates

        max_words = deps.settings.llm.max_tokens // 2
        system_text = ANSWER_SYSTEM.format(max_words=max_words)
        user_text = ANSWER_USER.format(context=context, question=question)
        prompt = build_messages(system_text, user_text)
        vision_messages = _vision_messages(
            system_text, user_text, state.get("hits", []), deps.image_root
        ) if supports_vision(deps.chat) else None
        try:
            if vision_messages is not None:
                try:
                    answer = deps.chat.invoke_messages(vision_messages)
                except Exception as exc:  # noqa: BLE001
                    log.warning("视觉生成失败，降级纯文本: %s", exc)
                    answer = deps.chat.invoke(prompt)
            else:
                answer = deps.chat.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            log.error("生成失败: %s", exc)
            answer = "抱歉，生成服务暂时不可用，请稍后重试。"
            updates.setdefault("errors", []).append(f"generate: {exc}")

        updates["answer"] = (answer or "").strip()
        updates["citations"] = [
            {"source": h["source"], "title": h.get("wiki_title") or h["title"],
             "chunk_id": h["chunk_id"], "score": h["score"],
             "image_path": h.get("image_path", "")}
            for h in state.get("hits", [])[:5]
        ]
        if deps.history:
            deps.history.append(state.get("session_id", "default"), question, updates["answer"])
        return updates

def node_verify(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """生成后校验：答案是否有依据（幻觉检查）。"""
    with _timed(state, "verify"):
        updates: dict[str, Any] = {"trace": state.get("trace", []),
                                   "timings": state.get("timings", {})}
        answer = state.get("answer", "")
        context = state.get("context", "")

        retries = int(state.get("generate_retry", 0))

        if not answer or "未" in answer[:12]:
            updates["grounded"] = False
            updates["generate_retry"] = retries + 1
            return updates

        if deps.offline or not deps.use_llm_judge:
            updates["grounded"] = bool(context.strip())
            if not updates["grounded"]:
                updates["generate_retry"] = retries + 1
            return updates

        try:
            raw = deps.chat.invoke(
                build_messages(VERIFY_SYSTEM,
                               VERIFY_USER.format(context=context[:4000], answer=answer))
            )
            match = re.search(r"\{.*\}", raw or "", re.S)
            data = json.loads(match.group(0)) if match else {}
            updates["grounded"] = bool(data.get("grounded", True))
            if not updates["grounded"]:
                reason = str(data.get("reason", ""))
                updates.setdefault("warnings", []).append(f"幻觉检查未通过: {reason}")
                updates["generate_retry"] = retries + 1
        except Exception as exc:  # noqa: BLE001
            log.warning("幻觉校验失败，按通过处理: %s", exc)
            updates["grounded"] = True
        return updates

def node_chitchat(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """闲聊直答：不进检索，省延迟省成本。"""
    with _timed(state, "chitchat"):
        return {
            "answer": "你好，我是企业知识库助手。可以问我产品、运维、故障相关的问题。",
            "grounded": True,
            "confidence": 1.0,
            "trace": state.get("trace", []),
            "timings": state.get("timings", {}),
        }

def node_clarify(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    """追问：问题缺实体时引导用户补充，避免无效检索。"""
    with _timed(state, "clarify"):
        return {
            "answer": "这个问题缺少关键对象（比如产品型号、模块名、报错码），"
                      "补充说明后我可以更准确地检索。",
            "grounded": False,
            "confidence": 0.0,
            "trace": state.get("trace", []),
            "timings": state.get("timings", {}),
        }
