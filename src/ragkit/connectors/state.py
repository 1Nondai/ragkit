"""同步状态：记录"每个 source 上次同步时的内容指纹"。

用 JSON 而不是 sqlite：状态只有 source → md5 一种映射，量级是文件数（千级），
JSON 足够，还方便人肉检查与手工修复。真到了十万级再换 sqlite，接口不变。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SyncStateStore:
    """source → 指纹 的持久化账本。"""

    path: Path
    _entries: dict[str, dict[str, str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> None:
        """从磁盘加载；文件不存在或损坏按空账本处理（首次同步/可重建状态）。"""
        if not self.path.exists():
            self._entries = {}
            return
        try:
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def get(self, source: str) -> dict[str, str] | None:
        """取某 source 的上次同步指纹；没同步过返回 None。"""
        return self._entries.get(source)

    def set(self, source: str, md5: str) -> None:
        self._entries[source] = {"md5": md5}

    def remove(self, source: str) -> None:
        self._entries.pop(source, None)

    def save(self) -> None:
        """落盘。父目录不存在时自动创建（state 路径可以放 data/ 下）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
