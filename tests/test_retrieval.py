"""检索层测试（全离线，用内存库 + 哈希向量）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.chunker import Chunk
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.rerank import KeywordOverlapReranker, NoopReranker
from ragkit.retrieval.service import RetrievalService

CORPUS = [
    ("Milvus 是一个开源的向量数据库，支持 HNSW、IVF_FLAT、DiskANN 等索引类型。", "overview.md"),
    ("安装 Milvus 需要 CPU 支持 SSE4.2、AVX、AVX2 或 AVX512 指令集。", "ops.md"),
    ("插入的数据存放在对象存储中，支持 MinIO、S3、GCS；元数据存放在 etcd。", "storage.md"),
    ("等离子体处理可用于半导体表面清洁、表面能改性与介电层钝化。", "tech.md"),
    ("单次插入的数据量不能超过 1024 MB，这是 gRPC 的限制。", "limit.md"),
]

def build_store() -> InMemoryStore:
    emb = HashingEmbedding(dim=128)
    store = InMemoryStore()
    store.ensure_collection(dim=128)
    chunks, vectors = [], []
    for i, (text, source) in enumerate(CORPUS):
        chunks.append(
            Chunk(text=text, metadata={
                "chunk_id": f"c{i}", "filename": source, "title": source,
                "category": "content", "source": source,
            })
        )
        vectors.append(emb.embed_query(text))
    store.upsert(chunks, vectors)
    return store

class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = build_store()
        cls.emb = HashingEmbedding(dim=128)
        cls.service = RetrievalService(store=cls.store, embeddings=cls.emb, top_k=3)

    def test_upsert_and_count(self):
        self.assertEqual(self.store.count(), len(CORPUS))

    def test_search_returns_hits_sorted_by_score(self):
        ctx = self.service.search("如何安装 Milvus")
        self.assertGreater(ctx.hit_count, 0)
        self.assertLessEqual(ctx.hit_count, 3)
        scores = [h.score for h in ctx.hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hit_carries_both_scores(self):
        ctx = self.service.search("etcd 元数据")
        self.assertTrue(any(h.chunk_id for h in ctx.hits))
        for hit in ctx.hits:
            self.assertIn("dense_score", hit.__dict__)
            self.assertIn("sparse_score", hit.__dict__)

    def test_metadata_filter(self):
        ctx = self.service.search("等离子体", filters={"filename": "tech.md"}, top_k=5)
        self.assertTrue(all(h.source == "tech.md" for h in ctx.hits))

    def test_empty_query_returns_nothing(self):
        ctx = self.service.search("   ")
        self.assertEqual(ctx.hit_count, 0)

    def test_context_text_has_citations(self):
        ctx = self.service.search("向量数据库")
        text = ctx.context_text()
        self.assertIn("来源:", text)
        self.assertIn("[1]", text)

    def test_rerank_changes_order_but_keeps_count(self):
        ctx = self.service.search("索引类型")
        before = [h.chunk_id for h in ctx.hits]
        after = [h.chunk_id for h in KeywordOverlapReranker().rerank("索引类型", ctx.hits, top_n=3)]
        self.assertEqual(len(before), len(after))
        self.assertEqual(set(before), set(after))

    def test_noop_reranker_truncates(self):
        ctx = self.service.search("Milvus", top_k=5)
        self.assertLessEqual(len(NoopReranker().rerank("Milvus", ctx.hits, top_n=2)), 2)

    def test_upsert_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.store.upsert([Chunk(text="x", metadata={"chunk_id": "z"})], [])

class TestRRF(unittest.TestCase):
    """单独验 RRF：融合的价值在于"救回单路高分但另一路平庸"的文档。"""

    def test_fusion_surfaces_document_strong_in_only_one_path(self):
        from ragkit.retrieval.protocol import SearchHit, _rrf_fuse

        def hit(cid: str, text: str) -> SearchHit:
            return SearchHit(chunk_id=cid, text=text, score=0.0, metadata={"chunk_id": cid})

        chunks = {cid: hit(cid, cid) for cid in ["a", "b", "c"]}
        dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        sparse = [("b", 0.6), ("c", 0.4), ("a", 0.01)]
        fused = [h.chunk_id for h in _rrf_fuse(chunks, dense, sparse, top_k=3)]
        self.assertEqual(fused[0], "b")
        self.assertLess(fused.index("a"), fused.index("c"))

    def test_fusion_respects_top_k(self):
        from ragkit.retrieval.protocol import SearchHit, _rrf_fuse

        chunks = {f"c{i}": SearchHit(chunk_id=f"c{i}", text="t", score=0.0) for i in range(10)}
        dense = [(f"c{i}", 1.0 - i / 10) for i in range(10)]
        sparse = [(f"c{i}", 0.5) for i in range(10)]
        self.assertEqual(len(_rrf_fuse(chunks, dense, sparse, top_k=4)), 4)

if __name__ == "__main__":
    unittest.main()
