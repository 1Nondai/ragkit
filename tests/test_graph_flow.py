"""图流程测试（离线、确定性）：路由分支、检索、生成、循环上限。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings
from ragkit.embeddings import HashingEmbedding
from ragkit.graph.graph import SimpleRunner, run
from ragkit.graph.nodes import Deps, HistoryStore, _keyword_grounding
from ragkit.graph.router import HybridRouter, KeywordRouter
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.llm import ExtractiveLLM
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.service import RetrievalService

CORPUS = {
    "产品手册.md": "# 产品手册\n\n## 安装\n\n安装 Milvus 需要 CPU 支持 AVX2 指令集，内存不低于 8 GB。\n",
    "故障排查.md": "# 故障排查\n\n## 错误码\n\n启动报 illegal instruction，说明 CPU 不支持所需 SIMD 指令集。\n",
}

def build_deps(tmp_dir: Path) -> Deps:
    store = InMemoryStore()
    embeddings = HashingEmbedding(dim=128)
    pipeline = IngestPipeline(store=store, embeddings=embeddings, chunk_size=200,
                              chunk_overlap=40, min_chunk_chars=10, batch_size=8)
    pipeline.run(tmp_dir, recreate=True)
    service = RetrievalService(store=store, embeddings=embeddings, top_k=3)
    settings = get_settings()
    settings.graph.max_rewrite = 1
    return Deps(service=service, chat=ExtractiveLLM(), router=HybridRouter(KeywordRouter()),
                settings=settings, history=HistoryStore())

class TestGraphFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        for name, text in CORPUS.items():
            (root / name).write_text(text, encoding="utf-8")
        cls.deps = build_deps(root)
        cls.runner = SimpleRunner(cls.deps)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_kb_question_retrieves_and_answers(self):
        state = run("安装 Milvus 需要什么 CPU 指令集？", self.deps, compiled=self.runner)
        self.assertEqual(state["route"], "kb")
        self.assertGreater(len(state["hits"]), 0)
        self.assertIn("AVX2", state["context"])
        self.assertTrue(state["answer"])

    def test_chitchat_short_circuits(self):
        state = run("你好", self.deps, compiled=self.runner)
        self.assertEqual(state["route"], "chitchat")
        self.assertEqual(state["hits"], [])
        self.assertNotIn("retrieve", state["trace"])

    def test_clarify_for_vague_question(self):
        state = run("这个是怎么办？", self.deps, compiled=self.runner)
        self.assertEqual(state["route"], "clarify")
        self.assertIn("关键对象", state["answer"])

    def test_rewrite_loop_is_bounded(self):
        state = run("完全无关的外部话题 xyzzy", self.deps, compiled=self.runner)
        self.assertLessEqual(state["rewrite_count"], self.deps.settings.graph.max_rewrite + 1)
        self.assertIn("generate", state["trace"])

    def test_grounded_consistency(self):
        """有 context 才允许判定 grounded；没有 context 必须给出"没检索到"的说法。"""
        state = run("量子引力理论的实验验证", self.deps, compiled=self.runner)
        self.assertTrue(state["answer"])
        if state.get("hits"):
            self.assertTrue(state["grounded"])
        else:
            self.assertFalse(state["grounded"])
            self.assertIn("未", state["answer"][:20])

    def test_trace_and_timings_recorded(self):
        state = run("启动报 illegal instruction 怎么办", self.deps, compiled=self.runner)
        for key in ("route", "rewrite", "retrieve", "grade", "generate", "verify"):
            self.assertIn(key, state["timings"]) if key in state["trace"] else None
        self.assertEqual(state["trace"][0], "route")

    def test_history_accumulates_per_session(self):
        run("启动报 illegal instruction 怎么办", self.deps, session_id="s1", compiled=self.runner)
        self.assertEqual(len(self.deps.history.get("s1")), 2)
        run("再问一次：illegal instruction", self.deps, session_id="s1", compiled=self.runner)
        self.assertEqual(len(self.deps.history.get("s1")), 4)
        self.assertEqual(self.deps.history.get("other-session"), [])

class TestLangGraphTermination(unittest.TestCase):
    """回归用：图必须会停下来。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.deps = build_deps(Path(cls._tmp.name))
        try:
            from ragkit.graph.graph import build_graph
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest("langgraph 未安装") from exc
        cls.app = build_graph(cls.deps)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_graph_terminates_when_nothing_is_found(self):
        state = run("完全查不到的外部话题 xyzzy", self.deps, compiled=self.app)
        self.assertEqual(state["trace"][-1], "verify")
        self.assertLessEqual(state.get("generate_retry", 0),
                             self.deps.settings.graph.max_generate_retry + 1)

class TestGroundingHeuristic(unittest.TestCase):
    def test_matching_terms_are_grounded(self):
        hits = [{"text": "Milvus 支持 HNSW 索引"}]
        self.assertGreaterEqual(_keyword_grounding("Milvus 支持哪些索引", hits), 0.3)

    def test_irrelevant_hits_are_not_grounded(self):
        hits = [{"text": "等离子体处理半导体表面"}]
        self.assertEqual(_keyword_grounding("Milvus 索引类型", hits), 0.0)

    def test_stopwords_are_ignored(self):
        hits = [{"text": "这是一个测试"}]
        self.assertEqual(_keyword_grounding("的 是 了", hits), 0.0)

if __name__ == "__main__":
    unittest.main()
