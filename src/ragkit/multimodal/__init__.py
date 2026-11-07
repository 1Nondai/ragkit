"""
多模态能力（OCR / 图片描述 / 图片落盘 / PDF 解析）。

只依赖标准库与 ragkit 自身；重依赖（rapidocr、PyMuPDF、httpx、PIL）
全部惰性 import，保证离线环境 `import ragkit.multimodal` 不炸。
"""

from ragkit.multimodal.captioner import CachedCaptioner, Captioner, LlmCaptioner, NullCaptioner
from ragkit.multimodal.image_store import ImageStore, StoredImage
from ragkit.multimodal.ocr_client import (
    DotsOcrClient,
    OcrBlock,
    OcrClient,
    RapidOcrClient,
    build_ocr,
)
from ragkit.multimodal.pdf_loader import PdfLoader

__all__ = [
    "OcrBlock", "OcrClient", "DotsOcrClient", "RapidOcrClient", "build_ocr",
    "ImageStore", "StoredImage",
    "Captioner", "LlmCaptioner", "NullCaptioner", "CachedCaptioner",
    "PdfLoader",
]
