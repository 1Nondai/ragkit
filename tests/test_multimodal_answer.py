"""图文回答测试（全离线）。"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.embeddings import HashingEmbedding
from ragkit.graph.nodes import Deps, _vision_messages, node_generate, node_retrieve
from ragkit.ingest.chunker import Chunk
from ragkit.llm import ExtractiveLLM, supports_vision
from ragkit.multimodal.image_store import image_data_url
from ragkit.retrieval.protocol import InMemoryStore
from ragkit.retrieval.service import RetrievalService

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

class FakeVisionChat:
    name = "fake-vision"

    def __init__(self) -> None:
        self.messages_seen: list[dict] | None = None
        self.invoke_calls = 0

    def invoke(self, prompt: str) -> str:
        self.invoke_calls += 1
        return "纯文本答案"

    def invoke_messages(self, messages: list[dict]) -> str:
        self.messages_seen = messages
        return "视觉答案"

class TestVisionMessages(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vision_"))
        (self.tmp / "images").mkdir()
        (self.tmp / "images" / "a.png").write_bytes(PNG)

    def test_builds_multimodal_message(self):
        hits = [{"image_path": "images/a.png"}, {"image_path": ""}]
        msgs = _vision_messages("sys", "user", hits, self.tmp)
        self.assertIsNotNone(msgs)
        self.assertEqual(msgs[0]["role"], "system")
        content = msgs[1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("user", content[0]["text"])
        images = [b for b in content if b["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_missing_file_dropped(self):
        hits = [{"image_path": "images/nope.png"}]
        self.assertIsNone(_vision_messages("s", "u", hits, self.tmp), "图读不出来就回纯文本")

    def test_no_root_or_no_hits(self):
        self.assertIsNone(_vision_messages("s", "u", [{"image_path": "images/a.png"}], None))
        self.assertIsNone(_vision_messages("s", "u", [], self.tmp))

    def test_dedup_same_image(self):
        hits = [{"image_path": "images/a.png"}, {"image_path": "images/a.png"}]
        msgs = _vision_messages("s", "u", hits, self.tmp)
        images = [b for b in msgs[1]["content"] if b["type"] == "image_url"]
        self.assertEqual(len(images), 1, "同一张图只带一次")

class TestNodeGenerate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="gen_"))
        (self.tmp / "images").mkdir()
        (self.tmp / "images" / "a.png").write_bytes(PNG)

    def _state(self, with_image: bool):
        hits = [{"chunk_id": "c1", "text": "正文", "score": 0.9, "source": "a.pdf",
                 "title": "图 3-1", "image_path": "images/a.png" if with_image else ""}]
        return {"question": "这个图讲了什么", "context": "[1] 来源: a.pdf\n正文", "hits": hits,
                "trace": [], "timings": {}}

    def test_vision_chat_gets_images(self):
        chat = FakeVisionChat()
        deps = Deps(service=None, chat=chat, image_root=self.tmp)  # type: ignore[arg-type]
        out = node_generate(self._state(True), deps)
        self.assertEqual(out["answer"], "视觉答案")
        self.assertIsNotNone(chat.messages_seen, "视觉模型必须走 invoke_messages")
        self.assertEqual(chat.invoke_calls, 0)

    def test_vision_failure_falls_back_to_text(self):
        """视觉调用抛异常（如纯文本模型收到图）→ 降级纯文本，答案不能丢。"""

        class BrokenVisionChat(FakeVisionChat):
            def invoke_messages(self, messages: list[dict]) -> str:
                self.messages_seen = messages
                raise RuntimeError("模型不支持图片输入")

        chat = BrokenVisionChat()
        deps = Deps(service=None, chat=chat, image_root=self.tmp)  # type: ignore[arg-type]
        out = node_generate(self._state(True), deps)
        self.assertEqual(out["answer"], "纯文本答案")
        self.assertEqual(chat.invoke_calls, 1, "降级后必须走一次纯文本 invoke")

    def test_offline_chat_falls_back_to_text(self):
        chat = ExtractiveLLM()
        self.assertFalse(supports_vision(chat))
        deps = Deps(service=None, chat=chat, image_root=self.tmp)  # type: ignore[arg-type]
        out = node_generate(self._state(True), deps)
        self.assertIsNotNone(out["answer"], "离线模式也要出答案（图片被忽略）")
        self.assertNotIn("视觉答案", out["answer"])

    def test_citations_carry_image_path(self):
        chat = FakeVisionChat()
        deps = Deps(service=None, chat=chat, image_root=self.tmp)  # type: ignore[arg-type]
        out = node_generate(self._state(True), deps)
        self.assertEqual(out["citations"][0]["image_path"], "images/a.png")

class TestNodeRetrievePassthrough(unittest.TestCase):
    def test_hits_carry_image_path(self):
        emb = HashingEmbedding(dim=128)
        store = InMemoryStore()
        store.ensure_collection(dim=128)
        chunk = Chunk(
            text="图中展示了 JobManager 与 TaskManager 的 RPC 交互时序。",
            metadata={"chunk_id": "c2", "filename": "ch3.pdf", "title": "图 3-2",
                      "source": "ch3.pdf", "category": "figure",
                      "modality": "image", "image_path": "images/3-2.png", "page": 12},
        )
        store.upsert([chunk], [emb.embed_query(chunk.text)])
        service = RetrievalService(store=store, embeddings=emb, top_k=3)
        deps = Deps(service=service, chat=ExtractiveLLM())  # type: ignore[arg-type]
        out = node_retrieve(
            {"question": "RPC 时序", "rewritten_queries": [], "trace": [], "timings": {}}, deps
        )
        hits = out["hits"]
        self.assertTrue(hits)
        self.assertEqual(hits[0]["modality"], "image")
        self.assertEqual(hits[0]["image_path"], "images/3-2.png")
        self.assertEqual(hits[0]["page"], 12)

class TestImageDataUrl(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dataurl_"))
        self.f = self.tmp / "a.png"
        self.f.write_bytes(PNG)

    def test_ok(self):
        url = image_data_url(self.f)
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_missing(self):
        self.assertIsNone(image_data_url(self.tmp / "nope.png"))

    def test_too_big(self):
        self.assertIsNone(image_data_url(self.f, max_bytes=10))

class TestExtractiveInvokeMessages(unittest.TestCase):
    def test_text_extracted_images_ignored(self):
        chat = ExtractiveLLM()
        answer = chat.invoke_messages([
            {"role": "system", "content": "【上下文】检索到了 Milvus 相关文档。\n【问题】Milvus 是什么"},
            {"role": "user", "content": [
                {"type": "text", "text": "Milvus 是什么？"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
            ]},
        ])
        self.assertIn("离线兜底模式", answer)

if __name__ == "__main__":
    unittest.main()
