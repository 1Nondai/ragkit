"""Wiki 层测试（全离线）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.ingest.chunker import Chunk
from ragkit.wiki.compiler import compile_pages, group_key
from ragkit.wiki.manager import WikiManager
from ragkit.wiki.router import WikiRouter, content_coverage, score_page
from ragkit.wiki.schema import WikiPage, safe_slug


def chunk(text: str, title: str, filename: str, cid: str) -> Chunk:
    return Chunk(text=text, metadata={
        "chunk_id": cid, "filename": filename, "title": title,
        "category": "content", "source": f"/data/{filename}", "doc_id": "d1",
    })

class TestCompiler(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            chunk("Milvus 是一个开源的向量数据库，支持 HNSW 索引。", "总览 -> 简介", "overview.md", "c1"),
            chunk("Milvus 是一个开源的向量数据库，支持 HNSW 索引。", "总览 -> 简介", "faq.md", "c2"),
            chunk("安装 Milvus 需要 CPU 支持 AVX2，内存不低于 8GB。", "总览 -> 安装", "ops.md", "c3"),
            chunk("启动报 illegal instruction 说明 CPU 不支持 SIMD 指令集。", "故障 -> 排查", "faq.md", "c4"),
        ]

    def test_group_by_leaf_title(self):
        self.assertEqual(group_key(self.chunks[0]), "简介")
        pages = compile_pages(self.chunks, today="2026-01-01")
        self.assertEqual({p.title for p in pages}, {"简介", "安装", "排查"})

    def test_body_dedup_and_sources(self):
        pages = {p.title: p for p in compile_pages(self.chunks, today="2026-01-01")}
        intro = pages["简介"]
        self.assertEqual(intro.body.count("开源的向量数据库"), 1, "重复段落应被去重")
        self.assertEqual(len(intro.sources), 2, "两份文档各贡献一个出处")

    def test_confidence_reflects_source_count(self):
        pages = {p.title: p for p in compile_pages(self.chunks, today="2026-01-01")}
        self.assertEqual(pages["简介"].confidence, "high")
        self.assertEqual(pages["安装"].confidence, "medium")

    def test_every_page_has_two_links(self):
        pages = compile_pages(self.chunks, today="2026-01-01")
        for page in pages:
            self.assertGreaterEqual(len(page.links), 2, f"{page.title} 是孤儿页")

    def test_frontmatter_roundtrip(self):
        page = WikiPage(slug="s", title="标题", body="正文", sources=["a.md#c1"],
                        created="2026-01-01", updated="2026-01-01",
                        tags=["content"], aliases=["Milvus"])
        parsed = WikiPage.from_markdown("s", page.to_markdown())
        self.assertEqual(parsed.title, "标题")
        self.assertEqual(parsed.sources, ["a.md#c1"])
        self.assertEqual(parsed.aliases, ["Milvus"])

    def test_slug_is_filesystem_safe(self):
        self.assertEqual(safe_slug("What is Milvus?"), "What-is-Milvus")
        self.assertNotIn("/", safe_slug("a/b c"))

class TestManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "wiki"
        self.manager = WikiManager(self.root)
        self.pages = compile_pages(
            [chunk("A 页面正文。相关说明。", "A", "a.md", "c1"),
             chunk("B 页面正文。相关说明。", "B", "b.md", "c2"),
             chunk("C 页面正文。相关说明。", "C", "c.md", "c3")],
            today="2026-01-01",
        )
        self.manager.build(self.pages)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_creates_index_schema_log(self):
        self.assertTrue((self.root / "index.md").exists())
        self.assertTrue((self.root / "SCHEMA.md").exists())
        self.assertTrue((self.root / "log.md").exists())
        self.assertEqual(len(self.manager.load()), 3)

    def test_index_lists_every_page(self):
        index = (self.root / "index.md").read_text(encoding="utf-8")
        for page in self.pages:
            self.assertIn(f"[[{page.title}]]", index)

    def test_lint_has_no_orphan_or_broken_link(self):
        findings = self.manager.lint()
        self.assertEqual(findings["orphan"], [])
        self.assertEqual(findings["broken_link"], [])
        self.assertEqual(findings["missing_frontmatter"], [])

    def test_stats_counts_hits(self):
        self.manager.record_query(True, "A", reset=True)
        self.manager.record_query(False)
        stats = self.manager.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["hits"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_chunk_page_index(self):
        index = self.manager.chunk_page_index()
        self.assertEqual(index.get("c1"), "A")

    def test_writeback_rejects_junk(self):
        self.assertIsNone(self.manager.writeback("q", "未在知识库中检索到相关内容", score=0.9))
        self.assertIsNone(self.manager.writeback("q", "不知道", score=0.9))
        self.assertIsNone(self.manager.writeback("q", "一段足够长的正常答案内容，用来测试分数门槛。", score=0.1))

    def test_writeback_writes_draft_once(self):
        answer = ("nlist 的经验值是 4 乘 sqrt(n)，n 是单个 segment 内的实体总数；"
                  "nprobe 需要在数据集上验证，它在精度和延迟之间做权衡。")
        path = self.manager.writeback("如何设置 nlist？", answer, score=0.9,
                                      sources=["a.md#c1"], min_chars=20)
        self.assertIsNotNone(path, "长度与分数都够时应写入草稿")
        self.assertIn("drafts", str(path))
        again = self.manager.writeback("如何设置 nlist？", answer, score=0.9,
                                       sources=["a.md#c1"], min_chars=20)
        self.assertIsNone(again)

class TestRouterGates(unittest.TestCase):
    @staticmethod
    def page(title: str, body: str = "", **kwargs) -> WikiPage:
        base = dict(slug=safe_slug(title), title=title, body=body,
                    sources=["a.md#c1"], created="2026-01-01", updated="2026-01-01")
        base.update(kwargs)
        return WikiPage(**base)

    def test_containment_beats_short_title_penalty(self):
        self.assertEqual(score_page(self.page("水位线"), "水位线是什么"), 1.0)

    def test_content_gate_blocks_generic_page(self):
        generic = self.page("Milvus 是什么", "Milvus 是开源的向量数据库。")
        router = WikiRouter([generic], threshold=0.6, min_coverage=0.2)
        decision = router.decide("Milvus 支持哪些索引类型？")
        self.assertFalse(decision.hit)
        self.assertEqual(decision.gate, "content")

    def test_low_confidence_blocked(self):
        p = self.page("索引类型", "支持 HNSW、IVF_FLAT、DiskANN 等索引类型。", confidence="low")
        router = WikiRouter([p], min_coverage=0.0)
        decision = router.decide("支持哪些索引类型")
        self.assertFalse(decision.hit)
        self.assertEqual(decision.gate, "confidence")

    def test_contested_blocked(self):
        p = self.page("索引类型", "支持 HNSW 等索引类型。", contested=True)
        router = WikiRouter([p], min_coverage=0.0)
        self.assertEqual(router.decide("支持哪些索引类型").gate, "contested")

    def test_stale_volatile_blocked(self):
        p = self.page("版本要求", "当前版本 2.5.4，要求 Python 3.8 以上。",
                      volatile=True, updated="2025-01-01")
        router = WikiRouter([p], min_coverage=0.0, stale_days=90, today="2026-01-01")
        self.assertEqual(router.decide("版本要求是什么").gate, "stale")

    def test_hit_returns_page_and_answer(self):
        p = self.page("索引类型", "支持 HNSW、IVF_FLAT、DiskANN 等索引类型。")
        router = WikiRouter([p], min_coverage=0.0)
        decision = router.decide("索引类型有哪些")
        self.assertTrue(decision.hit)
        self.assertIn("HNSW", WikiRouter.compose_answer(decision.page))
        self.assertIn("出处", WikiRouter.compose_answer(decision.page))

    def test_coverage_is_language_sensitive(self):
        """中文提问 + 英文正文 → 覆盖率必然低。这是本层最大的现实限制。"""
        p = self.page("Index types", "HNSW IVF_FLAT DiskANN are supported index types.")
        self.assertLess(content_coverage(p, "支持哪些索引类型"), 0.3)

class TestGraphIntegration(unittest.TestCase):
    def test_wiki_hit_short_circuits_retrieval(self):
        from ragkit.bootstrap import build_deps
        from ragkit.graph.graph import SimpleRunner, run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "wiki"
            manager = WikiManager(root)
            manager.build(compile_pages(
                [chunk("Milvus 支持 HNSW、IVF_FLAT、DiskANN 等索引类型。", "索引类型", "overview.md", "c1"),
                 chunk("安装 Milvus 需要 CPU 支持 AVX2。", "安装要求", "ops.md", "c2")],
                today="2026-01-01",
            ))
            deps = build_deps(prefer_memory=True)
            deps.settings.wiki.enabled = True
            deps.settings.wiki.min_coverage = 0.0
            deps.wiki_manager = manager
            from ragkit.wiki.router import WikiRouter
            deps.wiki_router = WikiRouter(manager.load(), min_coverage=0.0)

            state = run("索引类型有哪些", deps, compiled=SimpleRunner(deps))
            self.assertTrue(state["wiki_hit"])
            self.assertIn("索引", state["answer"])
            self.assertEqual(state["trace"], ["route", "wiki"])
            self.assertEqual(state["hits"], [])
            miss = run("完全无关的问题 xyzzy", deps, compiled=SimpleRunner(deps))
            self.assertFalse(miss["wiki_hit"])
            self.assertIn("retrieve", miss["trace"])

if __name__ == "__main__":
    unittest.main()
