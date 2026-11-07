"""图片描述（captioner）：给图片块补一段"它讲了什么"。"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

log = logging.getLogger("ragkit.multimodal.captioner")

CAPTION_PROMPT = (
    "用一句中文描述这张图/表格/公式图的内容与作用（例如：'展示了 JobManager 与 "
    "TaskManager 之间的 RPC 交互时序'）。只输出描述本身，不要任何前缀与解释。"
)

@runtime_checkable
class Captioner(Protocol):
    """描述器接口：一张图 → 一句中文描述（拿不到就返回空串）。"""
    def describe(self, image_bytes: bytes, context: str = "") -> str: ...

class NullCaptioner:
    """禁用/离线兜底：不生成描述。"""

    name = "null"

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        """生成描述；失败返回空串，调用方继续走“只有 OCR 文字”的降级。"""
        return ""

class LlmCaptioner:
    """调 OpenAI 兼容接口的视觉模型生成描述（httpx 直连，不依赖 langchain）。"""

    name = "llm"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        max_chars: int = 200,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_chars = max_chars
        self.timeout = timeout

    @staticmethod
    def _build_messages(image_bytes: bytes, context: str) -> list[dict]:
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        text = CAPTION_PROMPT
        if context:
            text += f"\n图片周边上下文：{context[:200]}"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        """生成描述；失败返回空串，调用方继续走“只有 OCR 文字”的降级。"""
        import httpx

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": self._build_messages(image_bytes, context),
                    "temperature": 0.1,
                    "max_tokens": 256,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 —— 描述失败不能中断入库
            log.warning("图片描述失败，回退空描述: %s", exc)
            return ""
        text = re.sub(r"\s+", " ", str(content or "")).strip()
        if len(text) > self.max_chars:
            log.info("描述超长截断: %d -> %d 字", len(text), self.max_chars)
            text = text[: self.max_chars]
        return text

class CachedCaptioner:
    """md5 缓存装饰：内存 dict + 可选磁盘文件（跨进程重启仍命中）。"""

    name = "cached"

    def __init__(self, inner: Captioner, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self._mem: dict[str, str] = {}
        self._dir = Path(cache_dir) if cache_dir else None

    @staticmethod
    def _md5(data: bytes) -> str:
        import hashlib
        return hashlib.md5(data).hexdigest()

    def describe(self, image_bytes: bytes, context: str = "") -> str:
        """生成描述；失败返回空串，调用方继续走“只有 OCR 文字”的降级。"""
        key = self._md5(image_bytes)
        if key in self._mem:
            return self._mem[key]
        if self._dir is not None:
            cache_file = self._dir / f"{key}.txt"
            if cache_file.exists():
                text = cache_file.read_text(encoding="utf-8")
                self._mem[key] = text
                return text
        text = self._inner.describe(image_bytes, context)
        if text:
            self._mem[key] = text
            if self._dir is not None:
                self._dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._dir / f"{key}.txt"
                cache_file.write_text(text, encoding="utf-8")
        return text
