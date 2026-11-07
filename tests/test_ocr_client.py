"""OCR 客户端测试（全离线：只测消息拼装、响应解析、注册表，不发真实请求）。"""

from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.multimodal.ocr_client import (
    LAYOUT_PROMPT,
    DotsOcrClient,
    RapidOcrClient,
    _parse_dots_json,
    build_ocr,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

class TestDotsMessages(unittest.TestCase):
    def test_build_messages_shape(self):
        messages = DotsOcrClient._build_messages(PNG_BYTES)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"), "必须是 data URL，vLLM 只认这种格式")
        self.assertEqual(url.split(",", 1)[1], base64.b64encode(PNG_BYTES).decode("ascii"))
        text = content[1]["text"]
        self.assertTrue(text.startswith("<|img|><|imgpad|><|endofimg|>"),
                        "占位符串不能丢：丢了 vLLM 会在图文交界插 \\n，版面判断偏移")
        self.assertIn(LAYOUT_PROMPT, text)

class TestParseDotsJson(unittest.TestCase):
    def test_clean_list(self):
        content = json.dumps([
            {"bbox": [1, 2, 3, 4], "category": "Title", "text": "# 标题"},
            {"bbox": [5, 6, 7, 8], "category": "Text", "text": "正文"},
        ])
        blocks = _parse_dots_json(content)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].category, "Title")
        self.assertEqual(blocks[0].bbox, (1.0, 2.0, 3.0, 4.0))

    def test_fenced_json(self):
        content = "```json\n[{\"category\": \"Text\", \"text\": \"a\"}]\n```"
        blocks = _parse_dots_json(content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "a")

    def test_result_wrapper(self):
        content = json.dumps({"result": [{"category": "Text", "text": "b"}]})
        blocks = _parse_dots_json(content)
        self.assertEqual(blocks[0].text, "b")

    def test_garbage_returns_empty(self):
        self.assertEqual(_parse_dots_json("模型突然开始闲聊，没有 JSON"), [])
        self.assertEqual(_parse_dots_json(""), [])
        self.assertEqual(_parse_dots_json('{"wrong": 1}'), [])

    def test_missing_fields_defaults(self):
        blocks = _parse_dots_json('[{"text": "x"}, 42, {"category": "Table", "bbox": "bad"}]')
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].category, "Text")
        self.assertIsNone(blocks[0].bbox)
        self.assertIsNone(blocks[1].bbox)

class TestBuildOcr(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(build_ocr("none"))

    def test_dots_with_kwargs(self):
        client = build_ocr("dots", base_url="http://x:6006/v1", model="m")
        self.assertIsInstance(client, DotsOcrClient)
        self.assertEqual(client.base_url, "http://x:6006/v1")
        self.assertEqual(client.model, "m")

    def test_rapidocr_lazy(self):
        client = build_ocr("rapidocr")
        self.assertIsInstance(client, RapidOcrClient)
        self.assertIsNone(client._engine, "构造时不加载模型（惰性），否则离线 import 会炸")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_ocr("tesseract")

class StaticOcrClient:
    """假 OCR：按固定规则吐块，供 pdf_loader 等下游模块测试复用。"""

    name = "static"

    def __init__(self, blocks: list | None = None) -> None:
        self.blocks = blocks or []
        self.calls: list[int] = []

    def parse_page(self, image_bytes: bytes, page: int) -> list:
        self.calls.append(page)
        return list(self.blocks)

if __name__ == "__main__":
    unittest.main()
