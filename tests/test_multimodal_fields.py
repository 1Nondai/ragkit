"""多模态字段测试（全离线）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.chunker import Chunk, chunk_sections
from ragkit.ingest.loader import LoadedDocument, Section
from ragkit.retrieval.milvus_store import MilvusStore
from ragkit.retrieval.protocol import InMemoryStore


class TestChunkDefaults(unittest.TestCase):
    def test_md_chunk_carries_default_triple(self):
        doc = LoadedDocument(
            source="C:/tmp/a.md", filename="a.md", filetype="md",
            sections=[Section(text="这是正文，长度要超过 min_chunk_chars 才不会被跳过。" * 3)],
        )
        chunks = chunk_sections(doc, chunk_size=800, chunk_overlap=120, min_chunk_chars=30)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertEqual(c.metadata.get("modality"), "text")
            self.assertEqual(c.metadata.get("image_path"), "")
            self.assertEqual(c.metadata.get("page"), 0)

class TestInMemoryRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self.emb = HashingEmbedding(dim=128)
        self.store = InMemoryStore()
        self.store.ensure_collection(dim=128)
        chunks = [
            Chunk(text="第三章 有界流处理与水位线，这是正文内容足够长的一段描述。",
                  metadata={"chunk_id": "c1", "filename": "ch3.md", "title": "3.1",
                            "category": "content", "source": "ch3.md",
                            "modality": "text", "image_path": "", "page": 0}),
            Chunk(text="图中展示了 JobManager 与 TaskManager 之间的 RPC 调用时序。",
                  metadata={"chunk_id": "c2", "filename": "ch3.pdf", "title": "图 3-2",
                            "category": "figure", "source": "ch3.pdf",
                            "modality": "image", "image_path": "images/3-2.png", "page": 12}),
        ]
        self.store.upsert(chunks, [self.emb.embed_query(c.text) for c in chunks])

    def test_hit_carries_triple(self):
        hits = self.store.search("JobManager 的 RPC 调用时序", self.emb.embed_query("RPC 时序"), top_k=5)
        image_hits = [h for h in hits if h.modality == "image"]
        self.assertTrue(image_hits, "图片块应该能被检索到")
        h = image_hits[0]
        self.assertEqual(h.image_path, "images/3-2.png")
        self.assertEqual(h.page, 12)

    def test_filter_by_modality(self):
        hits = self.store.search(
            "正文", self.emb.embed_query("正文"), top_k=5, filters={"modality": "image"}
        )
        self.assertTrue(all(h.modality == "image" for h in hits))

    def test_filter_by_page_int(self):
        hits = self.store.search(
            "正文", self.emb.embed_query("正文"), top_k=5, filters={"page": 12}
        )
        self.assertTrue(all(h.page == 12 for h in hits))

    def test_empty_filters_return_all(self):
        hits = self.store.search("正文", self.emb.embed_query("正文"), top_k=5, filters={})
        self.assertEqual(len(hits), 2)

class TestMilvusMappingNoPymilvus(unittest.TestCase):
    """不装 pymilvus 也能验证的映射逻辑（MilvusStore 是惰性 import）。"""

    def test_to_chunk_carries_triple(self):
        hit = {"entity": {
            "chunk_id": "c9", "doc_id": "d9", "filename": "b.pdf", "title": "图 4-1",
            "category": "figure", "source": "b.pdf", "text": "描述文字",
            "modality": "image", "image_path": "images/4-1.png", "page": 40,
        }}
        chunk = MilvusStore._to_chunk(hit)
        self.assertEqual(chunk.metadata["modality"], "image")
        self.assertEqual(chunk.metadata["image_path"], "images/4-1.png")
        self.assertEqual(chunk.metadata["page"], 40)

    def test_to_chunk_missing_fields_fallback(self):
        """旧数据没有新字段时按默认值兜底，不能抛异常。"""
        hit = {"entity": {"chunk_id": "c1", "text": "旧数据", "filename": "old.md"}}
        chunk = MilvusStore._to_chunk(hit)
        self.assertEqual(chunk.metadata["modality"], "text")
        self.assertEqual(chunk.metadata["image_path"], "")
        self.assertEqual(chunk.metadata["page"], 0)

    def test_build_expr_str_and_int(self):
        self.assertEqual(MilvusStore._build_expr(None), "")
        self.assertEqual(MilvusStore._build_expr({}), "")
        self.assertEqual(MilvusStore._build_expr({"modality": "image"}), 'modality == "image"')
        self.assertEqual(MilvusStore._build_expr({"page": 12}), "page == 12")
        self.assertEqual(
            MilvusStore._build_expr({"modality": "image", "page": 12}),
            'modality == "image" and page == 12',
        )

    def test_build_expr_escapes_quote(self):
        self.assertEqual(MilvusStore._build_expr({"title": 'a"b'}), 'title == "a\\"b"')

if __name__ == "__main__":
    unittest.main()
