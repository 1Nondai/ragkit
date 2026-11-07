"""检索抽象与内存实现。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragkit.ingest.chunker import Chunk


@dataclass
class SearchHit:

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank_fusion: str = "rrf"

    @property
    def source(self) -> str:
        """来源文件名（引用标记用）。"""
        return str(self.metadata.get("filename") or self.metadata.get("source") or "")

    @property
    def title(self) -> str:
        """标题路径（引用标记用）。"""
        return str(self.metadata.get("title") or "")

    @property
    def modality(self) -> str:
        """'text' | 'image'，缺失按 text 兜底（旧数据/旧 store 兼容）。"""
        return str(self.metadata.get("modality") or "text")

    @property
    def image_path(self) -> str:
        """图片块对应的图片相对路径；文本块为 ''。"""
        return str(self.metadata.get("image_path") or "")

    @property
    def page(self) -> int:
        """页码（1 起）；0 表示没有页码。"""
        return int(self.metadata.get("page") or 0)

    def citation(self) -> str:
        """给模型用的引用标记（同源去重时才不会全篇重复引用）。"""
        return f"[{self.source}{(' · ' + self.title) if self.title else ''}]"

@runtime_checkable
class VectorStore(Protocol):
    """向量库最小接口。"""

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None: ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        image_vectors: Sequence[Sequence[float]] | None = None,
    ) -> int: ...

    def search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[SearchHit]: ...

    def delete_by_source(self, source: str) -> int: ...

    def count(self) -> int: ...

    def drop(self) -> None: ...

@runtime_checkable
class ImageVectorStore(Protocol):
    """支持图片向量的向量库（图搜图）。不具备此能力的 store 直接跳过这一路。"""

    def search_by_image(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]: ...

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def _bm25_like(query: str, text: str, k1: float = 1.2, b: float = 0.75) -> float:
    """极简 BM25 近似（无语料统计，仅供离线打通链路）。"""
    import re

    q_terms = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    q_terms |= {ch for ch in (query or "") if "\u4e00" <= ch <= "\u9fff"}
    if not q_terms:
        return 0.0
    lowered = (text or "").lower()
    tf = sum(lowered.count(t) for t in q_terms)
    if tf == 0:
        return 0.0
    length_norm = 1 - b + b * (len(lowered) / 800)
    return (tf * (k1 + 1)) / (tf + k1 * length_norm)

class InMemoryStore:
    """内存向量库：稠密余弦 + 词面 BM25 近似 + RRF 融合。

    定位：测试、演示、CI。**不要**用它评估真实检索质量。
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._img_vectors: dict[str, list[float]] = {}
        self._dim: int | None = None

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """确保集合存在；recreate=True 时先删后建。"""
        if recreate:
            self.drop()
        self._dim = dim

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        image_vectors: Sequence[Sequence[float]] | None = None,
    ) -> int:
        """写入块与向量；chunk_id 相同即覆盖（幂等入库的基础）。"""
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量必须一致")
        if image_vectors is not None and len(image_vectors) != len(chunks):
            raise ValueError("image_vectors 与 chunks 数量必须一致")
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vec)
            if image_vectors is not None and chunk.metadata.get("image_path"):
                self._img_vectors[chunk.chunk_id] = list(image_vectors[i])
        return len(chunks)

    def count(self) -> int:
        """库内块数量。"""
        return len(self._chunks)

    def delete_by_source(self, source: str) -> int:
        """按来源删除块（增量同步：文档修改/删除时先清旧块）。"""
        stale = [
            cid for cid, c in self._chunks.items() if c.metadata.get("source") == source
        ]
        for cid in stale:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
            self._img_vectors.pop(cid, None)
        return len(stale)

    def drop(self) -> None:
        """清空数据。"""
        self._chunks.clear()
        self._vectors.clear()
        self._img_vectors.clear()

    def search_by_image(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """单路（图向量余弦）召回：只对带图向量的块打分。"""
        ranked: list[tuple[str, float]] = []
        for cid, vec in self._img_vectors.items():
            chunk = self._chunks.get(cid)
            if chunk is None or not self._match_filters(chunk.metadata, filters):
                continue
            ranked.append((cid, _cosine(query_vector, vec)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchHit(
                chunk_id=cid,
                text=self._chunks[cid].text,
                score=score,
                metadata=dict(self._chunks[cid].metadata),
                dense_score=score,
                sparse_score=0.0,
                rank_fusion="image",
            )
            for cid, score in ranked[:top_k]
        ]

    @staticmethod
    def _match_filters(metadata: dict[str, object], filters: dict[str, object] | None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                meta_val = metadata.get(key)
                meta_items = (
                    meta_val if isinstance(meta_val, (list, tuple, set))
                    else {str(meta_val)} if meta_val is not None else set()
                )
                if not (set(meta_items) & set(value)):
                    return False
                continue
            if metadata.get(key) != value:
                return False
        return True

    def search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        filters: dict[str, object] | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[SearchHit]:
        """单查询检索：文本两路融合 + 可选图片向量。"""
        dense_ranked: list[tuple[str, float]] = []
        sparse_ranked: list[tuple[str, float]] = []

        for cid, chunk in self._chunks.items():
            if not self._match_filters(chunk.metadata, filters):
                continue
            dense_ranked.append((cid, _cosine(query_vector, self._vectors[cid])))
            sparse_ranked.append((cid, _bm25_like(query_text, chunk.text)))

        dense_ranked.sort(key=lambda x: x[1], reverse=True)
        sparse_ranked.sort(key=lambda x: x[1], reverse=True)

        return _rrf_fuse(
            self._chunks, dense_ranked, sparse_ranked, top_k=top_k, rrf_k=rrf_k
        )

def _rrf_fuse(
    chunks: dict[str, Chunk],
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """互易等级融合（RRF）：按"排名"而不是"分数"合并两路结果。"""
    dense_scores = dict(dense_ranked)
    sparse_scores = dict(sparse_ranked)
    fused: dict[str, float] = {}

    for rank, (cid, _score) in enumerate(dense_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (cid, _score) in enumerate(sparse_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        SearchHit(
            chunk_id=cid,
            text=chunks[cid].text,
            score=score,
            metadata=dict(chunks[cid].metadata),
            dense_score=dense_scores.get(cid, 0.0),
            sparse_score=sparse_scores.get(cid, 0.0),
            rank_fusion="rrf",
        )
        for cid, score in ordered
    ]
