"""检索服务：把"向量库 + embedding + 重排"封成一个入口。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ragkit.embeddings import Embeddings
from ragkit.logging_setup import get_logger
from ragkit.retrieval.protocol import SearchHit, VectorStore
from ragkit.retrieval.rerank import Reranker, build_reranker

log = get_logger("retrieval.service")

@dataclass
class RetrievedContext:
    """一次检索的结果 + 给模型用的上下文文本。"""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    elapsed_ms: float = 0.0
    strategy: str = "hybrid+rrf"
    degraded: bool = False

    @property
    def hit_count(self) -> int:
        """命中条数。"""
        return len(self.hits)

    def context_text(self, max_chars: int = 6000) -> str:
        """拼上下文：带引用标记，且控制总长度。"""
        parts: list[str] = []
        used = 0
        for i, hit in enumerate(self.hits, start=1):
            block = f"[{i}] 来源: {hit.citation()}\n{hit.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def citations(self) -> list[dict[str, Any]]:
        """按 (来源, 标题) 去重后的引用列表。"""
        out = []
        seen = set()
        for hit in self.hits:
            key = (hit.source, hit.title)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "source": hit.source,
                    "title": hit.title,
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 4),
                }
            )
        return out

class RetrievalService:
    """检索门面。"""

    def __init__(
        self,
        store: VectorStore,
        embeddings: Embeddings,
        *,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        score_threshold: float = 0.0,
        rrf_k: int = 60,
        filter_category: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier
        self.score_threshold = score_threshold
        self.rrf_k = rrf_k
        self.filter_category = filter_category
        self.reranker = reranker or build_reranker()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ) -> RetrievedContext:
        """执行一次检索。"""
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return RetrievedContext(query=query)

        k = top_k or self.top_k
        filters = filters or (
            {"category": self.filter_category} if self.filter_category else None
        )

        vector = self.embeddings.embed_query(query)
        candidates_k = max(k * self.candidate_multiplier, k)

        hits = self.store.search(
            query_text=query,
            query_vector=vector,
            top_k=candidates_k,
            filters=filters,
            rrf_k=self.rrf_k,
        )

        if self.score_threshold > 0:
            hits = [h for h in hits if h.score >= self.score_threshold]

        before = [h.chunk_id for h in hits]
        hits = self.reranker.rerank(query, hits, top_n=k)
        degraded = bool(before) and not hits

        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            "检索完成: query=%r 命中=%d/%d 耗时=%.1fms", query[:40], len(hits), len(before), elapsed
        )
        return RetrievedContext(
            query=query, hits=hits, elapsed_ms=round(elapsed, 2), degraded=degraded
        )

    def search_many(self, queries: Sequence[str], **kwargs) -> list[RetrievedContext]:
        """多个 query 并行检索（改写的几个变体）。"""
        queries = list(queries)
        if len(queries) <= 1:
            return [self.search(q, **kwargs) for q in queries]
        workers = min(len(queries), 4)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recall") as pool:
            return list(pool.map(lambda q: self.search(q, **kwargs), queries))

    def search_image(
        self,
        image_bytes: bytes,
        top_k: int | None = None,
        *,
        filters: dict[str, object] | None = None,
    ) -> RetrievedContext:
        """图搜图：图片字节 → 图片向量 → 图向量路召回。"""
        k = top_k or self.top_k
        started = time.perf_counter()
        from ragkit.embeddings import ImageEmbeddings
        from ragkit.retrieval.protocol import ImageVectorStore

        if not isinstance(self.embeddings, ImageEmbeddings):
            log.warning("当前 embedding 不支持图片输入，图搜图不可用")
            return RetrievedContext(query="<image>", elapsed_ms=0.0, strategy="image",
                                    degraded=True)
        if not isinstance(self.store, ImageVectorStore):
            log.warning("当前向量库不支持图片向量，图搜图不可用")
            return RetrievedContext(query="<image>", elapsed_ms=0.0, strategy="image",
                                    degraded=True)
        try:
            vec = self.embeddings.embed_image(image_bytes)
            hits = self.store.search_by_image(vec, top_k=k, filters=filters)
        except Exception as exc:  # noqa: BLE001
            log.warning("图搜图失败: %s", exc)
            return RetrievedContext(query="<image>", elapsed_ms=0.0, strategy="image",
                                    degraded=True)
        return RetrievedContext(
            query="<image>", hits=hits,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            strategy="image",
        )
