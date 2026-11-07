"""OCR 客户端：把"一页图"变成"带版面的文字块"。"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger("ragkit.multimodal.ocr")

LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout "
    "element's bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', "
    "'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

@dataclass
class OcrBlock:
    """一页图里的一个版面块。"""

    category: str
    text: str
    bbox: tuple[float, float, float, float] | None = None

@runtime_checkable
class OcrClient(Protocol):
    """OCR 最小接口：一页图进去，一串版面块出来。"""

    name: str

    def parse_page(self, image_bytes: bytes, page: int) -> list[OcrBlock]:
        """image_bytes 必须是 PNG 字节（pdf_loader 保证）。page 仅用于日志与排错。"""
        ...

class DotsOcrClient:
    """调用 dots.ocr 服务。请求体形状与原项目 inference.py 保持一致。"""

    name = "dots"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6006/v1",
        model: str = "dots_ocr",
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 16384,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "0"
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @staticmethod
    def _build_messages(image_bytes: bytes) -> list[dict]:
        """纯函数：拼 OpenAI 多模态消息。抽出来是为了离线单测。"""
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{LAYOUT_PROMPT}"},
                ],
            }
        ]

    def parse_page(self, image_bytes: bytes, page: int) -> list[OcrBlock]:
        """调 dots.ocr 解析一页；服务异常返回空列表（由上游回落文本层）。"""
        import httpx

        payload = {
            "model": self.model,
            "messages": self._build_messages(image_bytes),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 —— OCR 服务挂了必须能降级，不能把入库拖崩
            log.warning("dots.ocr 调用失败（page=%s）: %s", page, exc)
            return []
        blocks = _parse_dots_json(content)
        if not blocks:
            log.warning("dots.ocr 返回无法解析（page=%s），回退文本层", page)
        return blocks

def _parse_dots_json(content: str) -> list[OcrBlock]:
    """解析 dots.ocr 的输出 JSON。"""
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("result") or data.get("blocks") or []
    if not isinstance(data, list):
        return []
    blocks: list[OcrBlock] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        try:
            bbox = tuple(float(v) for v in bbox[:4]) if isinstance(bbox, (list, tuple)) else None
        except (TypeError, ValueError):
            bbox = None
        blocks.append(OcrBlock(category=str(item.get("category", "Text")),
                               text=str(item.get("text", "")), bbox=bbox))
    return blocks

class RapidOcrClient:
    """rapidocr-onnxruntime 本地推理。只出文字、没有版面类别（全标 Text）。"""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def parse_page(self, image_bytes: bytes, page: int) -> list[OcrBlock]:
        """调 dots.ocr 解析一页；服务异常返回空列表（由上游回落文本层）。"""
        import io

        from PIL import Image

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result, _elapse = self._get_engine()(image)
        except Exception as exc:  # noqa: BLE001
            log.warning("RapidOCR 失败（page=%s）: %s", page, exc)
            return []
        blocks: list[OcrBlock] = []
        for box, text, _score in result or []:
            bbox = (float(box[0][0]), float(box[0][1]), float(box[2][0]), float(box[2][1]))
            blocks.append(OcrBlock(category="Text", text=str(text), bbox=bbox))
        return blocks

def build_ocr(provider: str, **kwargs) -> OcrClient | None:
    """按配置构造 OCR 客户端；provider='none' 返回 None（上游走文本层兜底）。"""
    if provider == "none":
        return None
    if provider == "dots":
        return DotsOcrClient(**kwargs)
    if provider == "rapidocr":
        return RapidOcrClient()
    raise ValueError(f"未知 OCR provider: {provider!r}（可选 none / rapidocr / dots）")
