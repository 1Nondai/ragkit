"""PDF 解析器：PDF → LoadedDocument（文本块 + 图片块）。"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path

from ragkit.ingest.loader import LoadedDocument, Section
from ragkit.multimodal.captioner import Captioner
from ragkit.multimodal.image_store import ImageStore
from ragkit.multimodal.ocr_client import OcrBlock, OcrClient

log = logging.getLogger("ragkit.multimodal.pdf_loader")

FIGURE_CATEGORIES = {"Picture", "Figure", "Image"}
TEXT_CATEGORIES = {"Text", "Title", "Section-header", "List-item",
                   "Caption", "Footnote", "Page-header", "Page-footer"}
SPECIAL_CATEGORIES = {"Table", "Formula"}

_TITLE_MAX_CHARS = 40

class PdfLoader:
    """把单个 PDF 解析成 LoadedDocument。

    blocks_cache：{页码: [OcrBlock]}。用于离线回放已有解析结果
    （例如 dots.ocr 提前跑好的 JSON），不必连 OCR 服务。
    """

    def __init__(
        self,
        ocr: OcrClient | None = None,
        *,
        image_store: ImageStore | None = None,
        captioner: Captioner | None = None,
        dpi: int = 150,
    ) -> None:
        self.ocr = ocr
        self.image_store = image_store
        self.captioner = captioner
        self.dpi = dpi

    def load(
        self, path: Path, *, blocks_cache: dict[int, list[OcrBlock]] | None = None
    ) -> LoadedDocument:
        """解析整个 PDF；blocks_cache 可离线回放已解析的版面块。

        用法：doc = PdfLoader(ocr=None, image_store=store).load(Path("a.pdf"))
              doc.sections  # 文本块 + 图片块（metadata 带 modality / image_path / page）
        """
        import pymupdf as fitz

        path = Path(path)
        doc = fitz.open(str(path))
        sections: list[Section] = []
        try:
            for i, page in enumerate(doc):
                page_no = i + 1
                blocks = (blocks_cache or {}).get(page_no)
                png: bytes | None = None
                if blocks is None and self.ocr is not None:
                    png = self._render(page)
                    blocks = self.ocr.parse_page(png, page_no)
                if blocks:
                    renderer: Callable[[], bytes]
                    if png is None:
                        renderer = lambda p=page: self._render(p)  # noqa: E731
                    else:
                        renderer = lambda _png=png: _png  # noqa: E731
                    made = self._blocks_to_sections(blocks, renderer, page_no)
                    if made:
                        sections.extend(made)
                        continue
                    log.warning("第 %d 页版面块为空，回退文本层", page_no)
                section = self._text_layer_section(page, page_no)
                if section is not None:
                    sections.append(section)
        finally:
            doc.close()
        return LoadedDocument(
            source=str(path), filename=path.name, filetype="pdf", sections=sections
        )

    def _render(self, page) -> bytes:
        import pymupdf as fitz

        pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72))
        return pix.tobytes("png")

    @staticmethod
    def _crop(png: bytes, bbox: tuple[float, float, float, float] | None) -> bytes | None:
        """从渲染图里按 bbox 裁出图片；越界钳到画布内，非法 bbox 返回 None。"""
        if bbox is None:
            return None
        from PIL import Image

        x1, y1, x2, y2 = (float(v) for v in bbox)
        try:
            image = Image.open(io.BytesIO(png))
            x1 = max(0.0, min(x1, image.width))
            x2 = max(0.0, min(x2, image.width))
            y1 = max(0.0, min(y1, image.height))
            y2 = max(0.0, min(y2, image.height))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            buf = io.BytesIO()
            image.crop((x1, y1, x2, y2)).save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001 —— 裁图失败只影响这一个图块
            log.warning("裁图失败: %s", exc)
            return None

    def _blocks_to_sections(
        self,
        blocks: list[OcrBlock],
        renderer: Callable[[], bytes],
        page_no: int,
    ) -> list[Section]:
        sections: list[Section] = []
        title_path: list[str] = []
        buffer: list[str] = []
        last_text = ""

        def flush_text() -> None:
            """把缓冲区里的正文块收成一个 Section（遇到标题/图片时调用）。"""
            nonlocal last_text
            text = "\n\n".join(b for b in buffer if b.strip()).strip()
            if text:
                sections.append(
                    Section(
                        text=text,
                        title_path=list(title_path),
                        level=len(title_path),
                        metadata={"modality": "text", "page": page_no, "category": "content"},
                    )
                )
                last_text = text
            buffer.clear()

        for b in blocks:
            category = b.category or "Text"
            if category in ("Title", "Section-header"):
                flush_text()
                clean = b.text.strip().lstrip("#").strip()
                if clean:
                    if category == "Title":
                        title_path = [clean]
                    elif title_path:
                        title_path = title_path[:1] + [clean]
                    else:
                        title_path = [clean]
            elif category in TEXT_CATEGORIES:
                if b.text.strip():
                    buffer.append(b.text.strip())
            elif category in SPECIAL_CATEGORIES:
                flush_text()
                if b.text.strip():
                    sections.append(
                        Section(
                            text=b.text.strip(),
                            title_path=list(title_path),
                            level=len(title_path),
                            metadata={
                                "modality": "text",
                                "page": page_no,
                                "category": "table" if category == "Table" else "formula",
                            },
                        )
                    )
            elif category in FIGURE_CATEGORIES:
                flush_text()
                img_section = self._figure_section(b, renderer, page_no, title_path, last_text)
                if img_section is not None:
                    sections.append(img_section)
        flush_text()
        return sections

    def _figure_section(
        self,
        block: OcrBlock,
        renderer: Callable[[], bytes],
        page_no: int,
        title_path: list[str],
        context: str,
    ) -> Section | None:
        """Picture 块 → 一个 modality=image 的 Section。

        裁不出图（无 image_store / 裁剪失败）也照样产出 section：
        文字信息（图注）不丢，只是没有 image_path —— 降级优先。
        """
        image_bytes = self._crop(renderer(), block.bbox)
        rel_path = ""
        if image_bytes and self.image_store is not None:
            try:
                rel_path = self.image_store.save(image_bytes).rel_path
            except Exception as exc:  # noqa: BLE001
                log.warning("图片落盘失败（page=%s）: %s", page_no, exc)
                image_bytes = None
        description = ""
        if image_bytes and self.captioner is not None:
            description = self.captioner.describe(image_bytes, context[-200:])

        caption = (block.text or "").strip()
        parts = [p for p in (f"[图片] {caption}" if caption else "[图片]", f"[描述] {description}") if p and p.strip()]
        text = "\n".join(parts)
        fig_title = caption[:30] or f"图 P{page_no}"
        return Section(
            text=text,
            title_path=title_path + [f"{fig_title} (P{page_no})"],
            level=len(title_path) + 1,
            metadata={
                "modality": "image",
                "image_path": rel_path,
                "page": page_no,
                "category": "figure",
            },
        )

    @staticmethod
    def _text_layer_section(page, page_no: int) -> Section | None:
        text = (page.get_text("text") or "").strip()
        if not text:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        title_path: list[str] = []
        if lines and len(lines[0]) <= _TITLE_MAX_CHARS and not lines[0].endswith(("。", ".", "！", "!")):
            title_path = [lines[0]]
            body_lines = lines[1:]
        else:
            body_lines = lines
        return Section(
            text="\n".join(body_lines),
            title_path=title_path,
            level=len(title_path),
            metadata={"modality": "text", "page": page_no, "category": "content"},
        )
