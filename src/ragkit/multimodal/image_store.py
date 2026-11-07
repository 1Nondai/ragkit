"""图片落盘：按内容 md5 命名，幂等去重。"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ragkit.multimodal.image_store")

MAX_DATA_URL_BYTES = 8 * 1024 * 1024

def image_data_url(abs_path: Path, max_bytes: int = MAX_DATA_URL_BYTES) -> str | None:
    """把本地图片读成 data URL（给视觉模型用）。"""
    path = Path(abs_path)
    try:
        if not path.exists() or path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "png" if suffix in ("", "png") else ("jpeg" if suffix in ("jpg", "jpeg") else suffix)
    return f"data:image/{mime};base64," + base64.b64encode(data).decode("ascii")

@dataclass
class StoredImage:
    """落盘结果：相对路径 + md5 + 是否首次写入（统计用）。"""
    rel_path: str
    md5: str
    is_new: bool

class ImageStore:
    """把图片字节落盘到 <root>/<subdir>/<md5><suffix>。"""

    def __init__(self, root: Path, subdir: str = "images") -> None:
        self.root = Path(root)
        self.subdir = subdir.strip("/")

    def save(self, image_bytes: bytes, suffix: str = ".png") -> StoredImage:
        """按内容 md5 存盘；同内容重复写入返回 is_new=False。

        用法：info = store.save(png_bytes)  # info.rel_path 形如 "images/<md5>.png"
        """
        if not suffix.startswith("."):
            suffix = "." + suffix
        md5 = hashlib.md5(image_bytes).hexdigest()
        rel_path = f"{self.subdir}/{md5}{suffix}"
        target = self.root / rel_path
        if target.exists():
            return StoredImage(rel_path=rel_path, md5=md5, is_new=False)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        tmp.write_bytes(image_bytes)
        os.replace(tmp, target)
        return StoredImage(rel_path=rel_path, md5=md5, is_new=True)
