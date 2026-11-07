"""入库流水线的多模态接线测试（全离线）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_ingest_pipeline
from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.loader import LoadedDocument, Section
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.multimodal.pdf_loader import PdfLoader
from ragkit.retrieval.protocol import InMemoryStore


class FakePdfLoader:
    def load(self, path: Path) -> LoadedDocument:
        return LoadedDocument(
            source=str(path), filename=path.name, filetype="pdf",
            sections=[Section(text="PDF 正文内容，长度超过三十个字符，足以通过 min_chunk_chars 过滤。",
                              title_path=["PDF 标题"], level=1,
                              metadata={"modality": "text", "page": 1})],
        )

class BrokenPdfLoader:
    def load(self, path: Path) -> LoadedDocument:
        raise RuntimeError("模拟解析失败")

class TestPipelineWithPdf(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pipeline_"))
        (self.tmp / "a.md").write_text(
            "# 标题\n\n这是 md 正文内容，长度超过三十个字符，足以通过 min_chunk_chars 过滤。\n",
            encoding="utf-8",
        )
        (self.tmp / "b.pdf").write_bytes(b"%PDF-1.4 fake")

    def _pipeline(self, pdf_loader) -> IngestPipeline:
        return IngestPipeline(
            store=InMemoryStore(), embeddings=HashingEmbedding(dim=128),
            pdf_loader=pdf_loader,
        )

    def test_load_collects_md_and_pdf(self):
        docs = self._pipeline(FakePdfLoader()).load(self.tmp)
        types = sorted(d.filetype for d in docs)
        self.assertEqual(types, ["md", "pdf"])

    def test_broken_pdf_skipped_md_kept(self):
        docs = self._pipeline(BrokenPdfLoader()).load(self.tmp)
        types = sorted(d.filetype for d in docs)
        self.assertEqual(types, ["md"], "PDF 挂了不影响 md 入库")

    def test_run_end_to_end_offline(self):
        pipeline = self._pipeline(FakePdfLoader())
        stats = pipeline.run(self.tmp, recreate=True)
        self.assertGreaterEqual(stats.chunks, 2, "md + PDF 都要出 chunk")
        self.assertEqual(stats.written, stats.chunks)

class TestBootstrapOffline(unittest.TestCase):
    def test_build_ingest_pipeline_no_key_no_ocr(self):
        """默认配置（provider=none、无 key）：装配不能崩，PDF 走文本层。"""
        pipeline = build_ingest_pipeline(prefer_memory=True)
        self.assertIsNotNone(pipeline.pdf_loader)
        self.assertIsInstance(pipeline.pdf_loader, PdfLoader)
        self.assertIsNone(pipeline.pdf_loader.ocr, "provider=none 时 OCR 应为 None")
        self.assertIsNotNone(pipeline.pdf_loader.image_store, "图片落盘目录要可配置")

if __name__ == "__main__":
    unittest.main()
