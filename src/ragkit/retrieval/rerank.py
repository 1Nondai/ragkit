"""重排（Rerank）：可选的第二阶段排序。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragkit.retrieval.protocol import SearchHit

logger = logging.getLogger("ragkit.retrieval.rerank")

@runtime_checkable
class Reranker(Protocol):
    """重排接口：输入候选与目标条数，输出重新排序后的列表。"""
    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]: ...

class NoopReranker:
    """不做重排（默认）。保留原顺序，只截断。"""

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        """对候选重新排序并截断到 top_n。"""
        return list(hits)[:top_n]

class KeywordOverlapReranker:
    """轻量重排：按问题与候选的词面重合度微调（零依赖，可离线跑）。"""

    def __init__(self, weight: float = 0.15) -> None:
        self.weight = weight

    @staticmethod
    def _terms(text: str) -> set:
        import re

        terms = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        terms |= {
            text[i : i + 2]
            for i in range(len(text or "") - 1)
            if all("\u4e00" <= ch <= "\u9fff" for ch in text[i : i + 2])
        }
        return {t for t in terms if t.strip()}

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        """对候选重新排序并截断到 top_n。"""
        q_terms = self._terms(query)
        scored = []
        for hit in hits:
            overlap = len(q_terms & self._terms(hit.text)) / (len(q_terms) or 1)
            scored.append((hit.score + self.weight * overlap, hit))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, hit in scored[:top_n]:
            hit.metadata["rerank_score"] = round(score, 4)
            out.append(hit)
        return out

class ApiReranker:
    """调用外部重排服务（OpenAI 兼容的 /rerank 或自定义 HTTP 接口）。

    这里只留接口与降级骨架：真实项目接 BGE-Reranker / Cohere Rerank /
    自建 TEI 服务时，把 HTTP 调用填进 `_call` 即可，其余逻辑不动。
    """

    def __init__(self, endpoint: str, model: str = "bge-reranker-v2-m3", timeout: float = 5.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def _call(self, query: str, texts: list[str]) -> list[float] | None:
        import httpx

        try:
            resp = httpx.post(
                self.endpoint,
                json={"model": self.model, "query": query, "documents": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "results" in data:
                return [float(r.get("relevance_score", 0.0)) for r in data["results"]]
            return [float(s) for s in data.get("scores", [])]
        except Exception as exc:  # noqa: BLE001
            logger.warning("重排服务调用失败，退回融合排序: %s", exc)
            return None

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        """对候选重新排序并截断到 top_n。"""
        hits = list(hits)
        scores = self._call(query, [h.text for h in hits])
        if scores is None or len(scores) != len(hits):
            return hits[:top_n]
        for hit, s in zip(hits, scores):
            hit.metadata["rerank_score"] = round(float(s), 4)
        ordered = sorted(hits, key=lambda h: h.metadata.get("rerank_score", h.score), reverse=True)
        return ordered[:top_n]

def build_reranker():
    """按配置构造重排器；未启用或异常时返回 Noop（保证链路不断）。"""
    from ragkit.config import get_settings

    s = get_settings().retrieval
    if not s.enable_rerank:
        return NoopReranker()
    if s.rerank_endpoint:
        return ApiReranker(endpoint=s.rerank_endpoint, model=s.rerank_model)
    return KeywordOverlapReranker()
