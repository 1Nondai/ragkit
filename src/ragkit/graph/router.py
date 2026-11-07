"""数据路由：决定这个问题该去哪儿。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragkit.logging_setup import get_logger

log = get_logger("graph.router")

CHITCHAT_PATTERNS = [
    r"^\s*(你好|您好|hi|hello|hey)\s*[!！。.?？]*\s*$",
    r"^\s*(谢谢|感谢|thanks|thx)\s*[!！。.?？]*\s*$",
    r"^\s*(你是谁|你叫什么|你能做什么|帮助|help)\s*[?？]?\s*$",
    r"^\s*(再见|拜拜|bye)\s*[!！。.?？]*\s*$",
]

CLARIFY_PATTERNS = [
    r"^\s*(这个|那个|它|这玩意).{0,6}(怎么办|如何|是什么原因|怎么弄)\s*[?？]?\s*$",
]

DEFAULT_KB_RULES: dict[str, list[str]] = {
    "product": ["产品", "规格", "型号", "参数", "报价", "选型"],
    "ops": ["运维", "部署", "安装", "监控", "扩容", "备份"],
    "trouble": ["报错", "故障", "异常", "失败", "排查", "报错码", "error"],
}

@dataclass
class RouteDecision:
    """路由结果。"""

    route: str
    kb_name: str | None = None
    reason: str = ""
    matched_rule: str | None = None
    confidence: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)

    def is_kb(self) -> bool:
        return self.route == "kb"

class KeywordRouter:
    """纯规则路由（零延迟、零成本、可解释）。"""

    def __init__(self, kb_rules: dict[str, list[str]] | None = None) -> None:
        self.kb_rules = kb_rules or DEFAULT_KB_RULES
        self._chitchat = [re.compile(p, re.IGNORECASE) for p in CHITCHAT_PATTERNS]
        self._clarify = [re.compile(p) for p in CLARIFY_PATTERNS]

    def decide(self, question: str) -> RouteDecision | None:
        """返回 None 表示规则没命中，交给上层（LLM）决定。"""
        q = (question or "").strip()

        for pattern in self._chitchat:
            if pattern.match(q):
                return RouteDecision("chitchat", reason="问候/客套，无需检索",
                                     matched_rule="chitchat")
        for pattern in self._clarify:
            if pattern.match(q):
                return RouteDecision("clarify", reason="问题缺关键实体，需追问",
                                     matched_rule="clarify")

        if self.kb_rules:
            hits: dict[str, int] = {}
            for kb, keywords in self.kb_rules.items():
                score = sum(1 for kw in keywords if kw in q)
                if score:
                    hits[kb] = score
            if hits:
                best = max(hits.items(), key=lambda x: x[1])
                return RouteDecision(
                    "kb", kb_name=best[0], reason=f"命中关键词库 {best[0]}",
                    matched_rule="keyword", confidence=min(1.0, 0.4 + 0.2 * best[1]),
                    extra={"scores": hits},
                )
        return None

class LlmRouter:
    """LLM 路由：规则覆盖不到的长尾交给模型（DeepSeek 等 OpenAI 兼容接口）。"""

    VALID = {"kb", "chitchat", "clarify"}

    def __init__(self, chat_model, *, default_kb: str | None = None) -> None:
        self.chat = chat_model
        self.default_kb = default_kb

    def decide(self, question: str, history: list[str] | None = None) -> RouteDecision:
        """返回路由结果；规则覆盖不到时交给 LLM，失败一律降级到 kb。"""
        from ragkit.graph.prompts import ROUTE_SYSTEM, ROUTE_USER, build_messages

        try:
            raw = self.chat.invoke(
                build_messages(ROUTE_SYSTEM, ROUTE_USER.format(question=question))
            )
            label = (raw or "").strip().lower()
            label = re.sub(r"[^a-z]", "", label)
            if label not in self.VALID:
                raise ValueError(f"未知路由标签: {label!r}")
            return RouteDecision(
                route=label,
                kb_name=self.default_kb if label == "kb" else None,
                reason="LLM 分类",
                matched_rule="llm",
                confidence=0.7,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 路由失败，安全降级到 kb: %s", exc)
            return RouteDecision("kb", kb_name=self.default_kb,
                                 reason=f"路由失败降级: {exc}", matched_rule="fallback",
                                 confidence=0.3)

class HybridRouter:
    """规则 → LLM 的两级路由。"""

    def __init__(self, keyword_router: KeywordRouter, llm_router: LlmRouter | None = None,
                 *, enable_llm: bool = True) -> None:
        self.keyword = keyword_router
        self.llm = llm_router
        self.enable_llm = enable_llm and llm_router is not None

    def decide(self, question: str, history: list[str] | None = None) -> RouteDecision:
        """返回路由结果；规则覆盖不到时交给 LLM，失败一律降级到 kb。"""
        hit = self.keyword.decide(question)
        if hit is not None:
            log.info("规则路由命中: %s -> %s", hit.matched_rule, hit.route)
            return hit
        if self.enable_llm and self.llm:
            return self.llm.decide(question, history)
        return RouteDecision("kb", reason="默认进知识库", matched_rule="default")
