"""PDF 解析器测试（离线：本地生成小 PDF，不依赖 OCR 服务）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf as fitz

from ragkit.ingest.chunker import chunk_sections
from ragkit.multimodal.image_store import ImageStore
from ragkit.multimodal.ocr_client import OcrBlock
from ragkit.multimodal.pdf_loader import PdfLoader


def make_pdf(path: Path, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for line in lines:
        page.insert_text((72, y), line, fontsize=12, fontname="china-s")
        y += 24
    doc.save(str(path))
    doc.close()

class CountingCaptioner:
    name = "counting"

    def __init__(self, reply: str = "一张示例图") -> None:
        self.reply = reply
        self.calls: list[bytes] = []

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        self.calls.append(image_bytes)
        return self.reply

class EmptyOcr:
    """模拟 OCR 服务挂掉：永远返回空。"""

    name = "empty"

    def parse_page(self, image_bytes: bytes, page: int) -> list:
        return []

class TestTextLayerFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfloader_"))
        self.pdf = self.tmp / "doc.pdf"
        make_pdf(self.pdf, ["第一章 概述", "这是第一段用于检索的正文。", "这是第二段正文，包含关键词 Milvus。"])

    def test_no_ocr_text_layer(self):
        doc = PdfLoader(ocr=None).load(self.pdf)
        self.assertEqual(doc.filetype, "pdf")
        self.assertEqual(doc.filename, "doc.pdf")
        self.assertTrue(doc.sections, "每页至少一个 section")
        s = doc.sections[0]
        self.assertEqual(s.title, "第一章 概述", "首行短句应识别为标题")
        self.assertIn("Milvus", s.text)
        self.assertEqual(s.metadata["modality"], "text")
        self.assertEqual(s.metadata["page"], 1)

    def test_ocr_failure_falls_back_to_text_layer(self):
        doc = PdfLoader(ocr=EmptyOcr()).load(self.pdf)
        self.assertTrue(doc.sections)
        self.assertIn("Milvus", doc.sections[0].text, "OCR 挂了必须回落文本层，不能丢内容")

class TestBlocksToSections(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfloader_"))
        self.pdf = self.tmp / "doc.pdf"
        make_pdf(self.pdf, ["第二章 有界流", "正文段落，讲 JobManager 与 TaskManager 的调度。", "更多正文。"])

    def test_picture_block_produces_image_section(self):
        store = ImageStore(root=self.tmp / "data")
        captioner = CountingCaptioner()
        blocks = [
            OcrBlock(category="Title", text="# 第二章 有界流"),
            OcrBlock(category="Text", text="正文段落，讲 JobManager 与 TaskManager 的调度。"),
            OcrBlock(category="Picture", text="", bbox=(60, 200, 220, 380)),
        ]
        doc = PdfLoader(ocr=None, image_store=store, captioner=captioner).load(
            self.pdf, blocks_cache={1: blocks}
        )
        images = [s for s in doc.sections if s.metadata.get("modality") == "image"]
        self.assertEqual(len(images), 1, "Picture 块必须产出 modality=image 的 section")
        img = images[0]
        self.assertTrue(img.metadata["image_path"].startswith("images/"))
        self.assertEqual(img.metadata["page"], 1)
        self.assertTrue((self.tmp / "data" / img.metadata["image_path"]).exists(), "图片文件要真实落盘")
        self.assertIn("[描述]", img.text)
        self.assertEqual(len(captioner.calls), 1, "同一张图只描述一次")

    def test_picture_without_store_degrades(self):
        blocks = [OcrBlock(category="Picture", text="图 2-1", bbox=(60, 200, 220, 380))]
        doc = PdfLoader(ocr=None, image_store=None, captioner=None).load(
            self.pdf, blocks_cache={1: blocks}
        )
        images = [s for s in doc.sections if s.metadata.get("modality") == "image"]
        self.assertEqual(len(images), 1, "没有 image_store 也要产出图片块")
        self.assertEqual(images[0].metadata["image_path"], "", "降级时 path 为空串")
        self.assertIn("图 2-1", images[0].text)

    def test_title_path_and_text_flow(self):
        blocks = [
            OcrBlock(category="Title", text="# 第二章 有界流"),
            OcrBlock(category="Section-header", text="## 2.1 水位线"),
            OcrBlock(category="Text", text="水位线用于处理乱序数据。"),
        ]
        doc = PdfLoader(ocr=None).load(self.pdf, blocks_cache={1: blocks})
        texts = [s for s in doc.sections if s.metadata.get("category") == "content"]
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].title, "第二章 有界流 -> 2.1 水位线")

class TestChunkerIntegration(unittest.TestCase):
    def test_image_section_survives_chunking(self):
        tmp = Path(tempfile.mkdtemp(prefix="pdfloader_"))
        pdf = tmp / "doc.pdf"
        make_pdf(pdf, ["第三章", "正文内容足够长的一段，保证文本块也正常产出。"] * 2)
        blocks = [OcrBlock(category="Picture", text="图 3-2", bbox=(60, 200, 220, 380))]
        doc = PdfLoader(ocr=None).load(pdf, blocks_cache={1: blocks})
        chunks = chunk_sections(doc, chunk_size=800, chunk_overlap=120, min_chunk_chars=30)
        image_chunks = [c for c in chunks if c.metadata.get("modality") == "image"]
        self.assertEqual(len(image_chunks), 1, "短图注不能因 min_chunk_chars 被丢弃")
        self.assertEqual(image_chunks[0].metadata["page"], 1)
        self.assertIn("(P1)", image_chunks[0].metadata["title"], "标题带页码，防跨页撞 chunk_id")

if __name__ == "__main__":
    unittest.main()
