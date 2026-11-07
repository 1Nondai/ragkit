"""图搜图测试（全离线，HashingEmbedding 的 embed_image 是确定性哈希）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding, OpenAICompatEmbedding
from ragkit.graph.nodes import Deps  # noqa: F401  (保持 import 一致性检查)
from ragkit.ingest.loader import LoadedDocument, Section
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.service import RetrievalService

PNG_A = b"\x89PNG\r\n\x1a\n" + b"a" * 300
PNG_B = b"\x89PNG\r\n\x1a\n" + b"b" * 300

class TestEmbedImage(unittest.TestCase):
    def setUp(self) -> None:
        self.emb = HashingEmbedding(dim=128)

    def test_deterministic_same_image(self):
        self.assertEqual(self.emb.embed_image(PNG_A), self.emb.embed_image(PNG_A))

    def test_different_image_different_vector(self):
        self.assertNotEqual(self.emb.embed_image(PNG_A), self.emb.embed_image(PNG_B))

    def test_normalized(self):
        vec = self.emb.embed_image(PNG_A)
        norm = sum(v * v for v in vec)
        self.assertAlmostEqual(norm, 1.0, places=6)

class TestMemoryStoreImageSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.emb = HashingEmbedding(dim=128)
        self.store = InMemoryStore()
        self.store.ensure_collection(dim=128)
        from ragkit.ingest.chunker import Chunk

        chunks = [
            Chunk(text="正文文本块，讲有界流处理。",
                  metadata={"chunk_id": "c1", "filename": "a.pdf", "title": "3.1",
                            "source": "a.pdf", "category": "content",
                            "modality": "text", "image_path": "", "page": 1}),
            Chunk(text="[图片]\n[描述] 架构图",
                  metadata={"chunk_id": "c2", "filename": "a.pdf", "title": "图 3-2 (P2)",
                            "source": "a.pdf", "category": "figure",
                            "modality": "image", "image_path": "images/f1.png", "page": 2}),
        ]
        self.store.upsert(
            chunks,
            [self.emb.embed_query(c.text) for c in chunks],
            image_vectors=[None, self.emb.embed_image(PNG_A)],
        )

    def test_search_by_image_hits_image_chunk(self):
        hits = self.store.search_by_image(self.emb.embed_image(PNG_A), top_k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, "c2", "同一张图必须命中自己的块")
        self.assertEqual(hits[0].image_path, "images/f1.png")
        self.assertEqual(hits[0].rank_fusion, "image")

    def test_text_chunk_excluded(self):
        hits = self.store.search_by_image(self.emb.embed_image(PNG_A), top_k=10)
        ids = [h.chunk_id for h in hits]
        self.assertNotIn("c1", ids, "文本块没有图向量，不能出现在图检索结果里")

class TestServiceImageSearch(unittest.TestCase):
    def test_works_with_hashing(self):
        store = InMemoryStore()
        store.ensure_collection(dim=128)
        from ragkit.ingest.chunker import Chunk

        chunk = Chunk(text="[图片] 图 1", metadata={
            "chunk_id": "c1", "filename": "a.pdf", "title": "图 1", "source": "a.pdf",
            "modality": "image", "image_path": "images/f.png", "page": 1,
        })
        emb = HashingEmbedding(dim=128)
        store.upsert([chunk], [emb.embed_query(chunk.text)],
                     image_vectors=[emb.embed_image(PNG_A)])
        service = RetrievalService(store=store, embeddings=emb, top_k=3)
        ctx = service.search_image(PNG_A)
        self.assertFalse(ctx.degraded)
        self.assertEqual(ctx.strategy, "image")
        self.assertEqual(ctx.hit_count, 1)

    def test_degraded_when_embedding_not_image_capable(self):
        store = InMemoryStore()
        store.ensure_collection(dim=1024)
        emb = OpenAICompatEmbedding(dim=1024)
        service = RetrievalService(store=store, embeddings=emb, top_k=3)
        ctx = service.search_image(PNG_A)
        self.assertTrue(ctx.degraded)
        self.assertEqual(ctx.hit_count, 0)

class FakePdfLoader:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "images" / "f1.png").write_bytes(PNG_A)

    def load(self, path: Path) -> LoadedDocument:
        return LoadedDocument(
            source=str(path), filename=path.name, filetype="pdf",
            sections=[Section(
                text="[图片] 架构图\n[描述] 展示调度流程",
                title_path=["图 1 (P1)"], level=1,
                metadata={"modality": "image", "image_path": "images/f1.png",
                          "page": 1, "category": "figure"},
            )],
        )

class TestPipelineImageVectors(unittest.TestCase):
    def test_image_vectors_flow_and_search(self):
        tmp = Path(tempfile.mkdtemp(prefix="imgvec_"))
        (tmp / "doc.pdf").write_bytes(b"%PDF fake")
        emb = HashingEmbedding(dim=128)
        store = InMemoryStore()
        pipeline = IngestPipeline(
            store=store, embeddings=emb, pdf_loader=FakePdfLoader(tmp), image_root=tmp,
        )
        stats = pipeline.run(tmp, recreate=True)
        self.assertEqual(stats.chunks, 1)
        self.assertEqual(stats.image_vectors, 1, "入库统计要记录图片向量数")

        service = RetrievalService(store=store, embeddings=emb, top_k=3)
        ctx = service.search_image(PNG_A)
        self.assertEqual(ctx.hit_count, 1, "入库后拿原图应能搜回自己的块")
        self.assertEqual(ctx.hits[0].image_path, "images/f1.png")

if __name__ == "__main__":
    unittest.main()
