"""观测层测试：降级路径（未启用/缺 key/SDK 缺失）、FakeSink 的 span 与分数记录、ObservableChat 的 generation 记录。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import reload_settings
from ragkit.embeddings import HashingEmbedding
from ragkit.graph.graph import run
from ragkit.graph.nodes import Deps, HistoryStore
from ragkit.graph.router import HybridRouter, KeywordRouter
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.llm import ExtractiveLLM
from ragkit.obs.langfuse_sink import ObservableChat, build_trace_sink
from ragkit.obs.protocol import NoopSink
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.service import RetrievalService

CORPUS = {
    "产品手册.md": "# 产品手册\n\n## 安装\n\n安装 Milvus 需要 CPU 支持 AVX2 指令集，内存不低于 8 GB。\n",
}

class FakeGen:
    """generation_span yield 出来的记录器。"""

    def __init__(self, sink: FakeSink, name: str, input_: object, model: object) -> None:
        self._sink = sink
        self.name = name
        self.input = input_
        self.model = model

    def update(self, **kwargs: object) -> None:
        self._sink.last_gen_update = {"name": self.name, "input": self.input,
                                      "model": self.model, **kwargs}

class FakeSink:
    """TraceSink 的内存实现：把所有调用记成列表供断言。"""

    active = True

    def __init__(self) -> None:
        self.request_starts = 0
        self.node_spans: list[str] = []
        self.generations = 0
        self.results: list[dict] = []
        self.errors: list[BaseException] = []
        self.flushes = 0
        self.last_gen_update: dict | None = None

    @contextmanager
    def request_span(self, state: dict):
        self.request_starts += 1
        yield self

    @contextmanager
    def node_span(self, name: str):
        self.node_spans.append(name)
        yield None

    @contextmanager
    def generation_span(self, *, name: str, input, model, metadata=None):
        self.generations += 1
        yield FakeGen(self, name, input, model)

    def record_result(self, state: dict) -> None:
        self.results.append({"answer": state.get("answer", ""),
                             "confidence": state.get("confidence", 0.0)})

    def record_error(self, exc: BaseException) -> None:
        self.errors.append(exc)

    def flush(self) -> None:
        self.flushes += 1

def build_offline_deps(tmp_dir: Path, sink: object | None = None) -> Deps:
    """最小离线 Deps：内存库 + 哈希向量 + 抽句式生成（不依赖任何外部服务）。"""
    store = InMemoryStore()
    embeddings = HashingEmbedding(dim=128)
    pipeline = IngestPipeline(store=store, embeddings=embeddings, chunk_size=200,
                              chunk_overlap=40, min_chunk_chars=10, batch_size=8)
    pipeline.run(tmp_dir, recreate=True)
    service = RetrievalService(store=store, embeddings=embeddings, top_k=3)
    return Deps(service=service, chat=ExtractiveLLM(), router=HybridRouter(KeywordRouter()),
                history=HistoryStore(), trace_sink=sink)

class TestBuildTraceSink(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        sink = build_trace_sink()
        self.assertIsInstance(sink, NoopSink)
        self.assertFalse(sink.active)

    def test_enabled_without_keys_degrades(self) -> None:
        from ragkit.config import Settings

        s = Settings()
        s.obs.enabled = True
        s.obs.public_key = None
        s.obs.secret_key = None
        self.assertIsInstance(build_trace_sink(s), NoopSink)

    def test_sdk_missing_degrades(self) -> None:
        """SDK 缺失时降级为 NoopSink（sys.modules 注入 None 模拟未安装）。"""
        from ragkit.config import Settings

        s = Settings()
        s.obs.enabled = True
        s.obs.public_key = "pk-test"
        s.obs.secret_key = "sk-test"
        saved = sys.modules.get("langfuse")
        sys.modules["langfuse"] = None
        try:
            sink = build_trace_sink(s)
        finally:
            if saved is None:
                sys.modules.pop("langfuse", None)
            else:
                sys.modules["langfuse"] = saved
        self.assertIsInstance(sink, NoopSink)

class TestRunWithSink(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_records_spans_and_result(self) -> None:
        fake = FakeSink()
        deps = build_offline_deps(self.tmp, sink=fake)
        state = run("安装 Milvus 需要什么配置", deps)
        self.assertEqual(fake.request_starts, 1)
        for name in ("route", "wiki", "rewrite", "retrieve", "grade", "generate", "verify"):
            self.assertIn(name, fake.node_spans, f"缺少节点 span: {name}")
        self.assertEqual(len(fake.results), 1)
        self.assertEqual(fake.flushes, 0)
        self.assertTrue(state.get("answer") or state.get("hits"))

    def test_run_without_sink_unchanged(self) -> None:
        deps = build_offline_deps(self.tmp, sink=None)
        state = run("安装 Milvus 需要什么配置", deps)
        self.assertTrue(state.get("answer") or state.get("hits"))
        self.assertEqual(len(FakeSink().results), 0)

    def test_noop_sink_zero_overhead_path(self) -> None:
        deps = build_offline_deps(self.tmp, sink=NoopSink())
        state = run("安装 Milvus 需要什么配置", deps)
        self.assertTrue(state.get("answer") or state.get("hits"))

class TestObservableChat(unittest.TestCase):
    def test_invoke_records_generation(self) -> None:
        fake = FakeSink()

        class StubChat:
            model = "stub-model"

            def invoke(self, prompt: str) -> str:
                return f"echo:{prompt}"

            def invoke_messages(self, messages: list) -> str:
                return "ok"

        chat = ObservableChat(StubChat(), fake)
        out = chat.invoke("你好")
        self.assertEqual(out, "echo:你好")
        self.assertEqual(fake.generations, 1)
        self.assertIsNotNone(fake.last_gen_update)
        self.assertEqual(fake.last_gen_update["output"], "echo:你好")
        self.assertEqual(fake.last_gen_update["model"], "stub-model")
        self.assertIn("latency_ms", fake.last_gen_update["metadata"])

    def test_attribute_passthrough(self) -> None:
        fake = FakeSink()

        class StubChat:
            model = "stub"
            base_url = "http://example"

            def invoke(self, prompt: str) -> str:
                return prompt

        chat = ObservableChat(StubChat(), fake)
        self.assertEqual(chat.base_url, "http://example")

class TestObsConfig(unittest.TestCase):
    def test_env_override(self) -> None:
        import os

        os.environ["RAG_OBS__ENABLED"] = "true"
        os.environ["RAG_OBS__SAMPLE_RATE"] = "0.5"
        try:
            s = reload_settings()
            self.assertTrue(s.obs.enabled)
            self.assertAlmostEqual(s.obs.sample_rate, 0.5)
        finally:
            os.environ.pop("RAG_OBS__ENABLED", None)
            os.environ.pop("RAG_OBS__SAMPLE_RATE", None)
            reload_settings()

    def test_redacted_hides_keys(self) -> None:
        from ragkit.config import redacted

        snap = redacted()
        self.assertIn("obs", snap)
        self.assertIsInstance(snap["obs"]["keys_set"], bool)
        self.assertNotIn("public_key", snap["obs"])

if __name__ == "__main__":
    unittest.main()
