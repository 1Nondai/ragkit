"""权限过滤（acl）测试：入库打标 + 检索前置过滤 + Milvus 表达式。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from typing import cast

from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.loader import parse_front_matter
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.retrieval.milvus_store import MilvusStore
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.rerank import NoopReranker
from ragkit.retrieval.service import RetrievalService

_S = 8

DOCS = [
    ("# 财务制度\n\n差旅报销标准是每天 300 元。" * _S, {"acl": ["finance"]}, "finance.md"),
    ("# 运维手册\n\n重启 Milvus 前必须先备份 MinIO 数据。" * _S, {"acl": ["ops"]}, "ops.md"),
    ("# 产品介绍\n\nMilvus 是一个开源向量数据库。" * _S, {"acl": ["public"]}, "intro.md"),
    ("# 公开 FAQ\n\n单次插入上限是 1,024 MB。" * _S, {}, "faq.md"),
]

def _front_matter_text(body: str, meta: dict) -> str:
    fm = ""
    if meta.get("acl"):
        fm = f"---\nacl: {','.join(meta['acl'])}\n---\n\n"
    return fm + body

def _ingest(tmp: Path, default_acl: str = "public") -> InMemoryStore:
    store = InMemoryStore()
    pipeline = IngestPipeline(store=store, embeddings=HashingEmbedding(dim=32),
                              default_acl=default_acl)
    for body, meta, name in DOCS:
        (tmp / name).write_text(_front_matter_text(body, meta), encoding="utf-8")
    pipeline.run(tmp)
    return store

class TestFrontMatter(unittest.TestCase):
    def test_parse_acl(self) -> None:
        meta, text = parse_front_matter("---\nacl: finance,ops\n---\n\n正文")
        self.assertEqual(meta["acl"], "finance,ops")
        self.assertTrue(text.startswith("正文"))

    def test_no_front_matter(self) -> None:
        meta, text = parse_front_matter("# 标题\n\n正文")
        self.assertEqual(meta, {})
        self.assertIn("正文", text)

class TestAclIngest(unittest.TestCase):
    def test_default_acl_applied(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = _ingest(Path(tmp))
            acls = {tuple(cast(list, c.metadata.get("acl") or [])) for c in store._chunks.values()}
            self.assertIn(("public",), acls)
            self.assertIn(("finance",), acls)

    def test_doc_acl_overrides_default(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = _ingest(Path(tmp))
            finance = [
                c for c in store._chunks.values()
                if c.metadata.get("filename") == "finance.md"
            ]
            self.assertTrue(finance)
            self.assertTrue(all(c.metadata.get("acl") == ["finance"] for c in finance))

class TestAclSearch(unittest.TestCase):
    def _service(self, tmp: Path) -> RetrievalService:
        store = _ingest(tmp)
        return RetrievalService(store=store, embeddings=HashingEmbedding(dim=32),
                                reranker=NoopReranker())

    def test_filter_by_group(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            ctx = service.search("报销标准", acl=["finance"])
            self.assertTrue(ctx.hits)
            for h in ctx.hits:
                acl_meta = h.metadata.get("acl") or []
                self.assertIn("finance", list(cast(list, acl_meta)))

    def test_no_group_no_filter(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            ctx_no_acl = service.search("重启 Milvus")
            ctx_all = service.search("重启 Milvus", acl=["ops"])
            self.assertTrue(ctx_no_acl.hits)
            self.assertTrue(ctx_all.hits)

    def test_group_without_access_sees_nothing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            ctx = service.search("差旅报销", acl=["ops"])
            for h in ctx.hits:
                self.assertNotEqual(h.metadata.get("filename"), "finance.md")
            ctx_all = service.search("差旅报销")
            self.assertTrue(
                any(h.metadata.get("filename") == "finance.md" for h in ctx_all.hits),
                "不带 acl 时应能召回 finance 文档（对照组）",
            )

class TestMilvusExpr(unittest.TestCase):
    def test_list_value_becomes_json_contains_any(self) -> None:
        expr = MilvusStore._build_expr({"acl": ["public", "ops"], "kb": "product"})
        self.assertIn('json_contains_any(acl, ["public", "ops"])', expr)
        self.assertIn('kb == "product"', expr)
        self.assertIn(" and ", expr)

    def test_quote_escaping(self) -> None:
        expr = MilvusStore._build_expr({"acl": ['a"b']})
        self.assertIn(r'\"', expr)

if __name__ == "__main__":
    unittest.main()
