"""图片描述测试（全离线：假 inner 测缓存语义，纯函数测消息拼装）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.multimodal.captioner import (
    CAPTION_PROMPT,
    CachedCaptioner,
    LlmCaptioner,
    NullCaptioner,
)

A = b"\x89PNG\r\n\x1a\n" + b"a" * 100
B = b"\x89PNG\r\n\x1a\n" + b"b" * 100

class CountingCaptioner:
    """假实现：记录调用次数与参数，返回固定文本。"""

    name = "counting"

    def __init__(self, reply: str = "这是一张架构图") -> None:
        self.reply = reply
        self.calls: list[tuple[bytes, str]] = []

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        self.calls.append((image_bytes, context))
        return self.reply

class TestNullCaptioner(unittest.TestCase):
    def test_always_empty(self):
        self.assertEqual(NullCaptioner().describe(A, "ctx"), "")

class TestCachedCaptioner(unittest.TestCase):
    def test_memory_cache_hits(self):
        inner = CountingCaptioner()
        cached = CachedCaptioner(inner)
        self.assertEqual(cached.describe(A), "这是一张架构图")
        self.assertEqual(cached.describe(A), "这是一张架构图")
        self.assertEqual(len(inner.calls), 1, "同图第二次必须命中缓存")
        cached.describe(B)
        self.assertEqual(len(inner.calls), 2, "不同图要重新描述")

    def test_disk_cache_survives_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            inner = CountingCaptioner()
            c1 = CachedCaptioner(inner, cache_dir=Path(tmp))
            self.assertEqual(c1.describe(A), "这是一张架构图")
            c2 = CachedCaptioner(inner, cache_dir=Path(tmp))
            self.assertEqual(c2.describe(A), "这是一张架构图")
            self.assertEqual(len(inner.calls), 1, "磁盘缓存跨实例必须命中")

    def test_empty_reply_not_cached(self):
        inner = CountingCaptioner(reply="")
        cached = CachedCaptioner(inner)
        self.assertEqual(cached.describe(A), "")
        self.assertEqual(cached.describe(A), "")
        self.assertEqual(len(inner.calls), 2, "空描述不能进缓存，否则降级结果被永久记住")

class TestLlmCaptionerPure(unittest.TestCase):
    def test_build_messages(self):
        messages = LlmCaptioner._build_messages(A, "3.2 节 有界流")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        text = content[1]["text"]
        self.assertIn(CAPTION_PROMPT, text)
        self.assertIn("3.2 节 有界流", text, "周边上下文要注入 prompt")

    def test_no_context_ok(self):
        text = LlmCaptioner._build_messages(A, "")[0]["content"][1]["text"]
        self.assertNotIn("周边上下文", text)
