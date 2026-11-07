"""本地目录连接器：用内容 md5（而不是 mtime）检测变更。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragkit.connectors.base import ChangeRecord, ChangeType
from ragkit.connectors.state import SyncStateStore
from ragkit.ingest.loader import SUPPORTED_SUFFIX, LoadedDocument, load_file


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

class LocalDirConnector:
    """把一个本地目录包装成数据源。目录里文件的增删改都能被检出来。"""

    name = "local_dir"

    def __init__(self, root: Path, state: SyncStateStore,
                 *, suffixes: set[str] | None = None) -> None:
        self.root = Path(root)
        self.state = state
        self.suffixes = {s.lower() for s in (suffixes or SUPPORTED_SUFFIX)}

    def _current_files(self) -> dict[str, str]:
        """当前目录快照：绝对路径 → 内容 md5（只看支持的后缀）。"""
        out: dict[str, str] = {}
        if not self.root.exists():
            return out
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.suffixes:
                out[str(p.resolve())] = _md5(p.read_bytes())
        return out

    def list_changes(self) -> list[ChangeRecord]:
        """状态账本 vs 当前快照，产出增/删/改三类变更。"""
        current = self._current_files()
        records: list[ChangeRecord] = []
        for source, md5 in current.items():
            prev = self.state.get(source)
            if prev is None:
                records.append(ChangeRecord(source, Path(source).name, ChangeType.ADDED))
            elif prev.get("md5") != md5:
                records.append(ChangeRecord(source, Path(source).name, ChangeType.MODIFIED))
        for source in self.state._entries:
            if source not in current:
                records.append(
                    ChangeRecord(source, Path(source).name, ChangeType.DELETED)
                )
        return records

    def fetch(self, source: str) -> LoadedDocument:
        """按 source 加载单个文档（交给上层切块入库）。"""
        return load_file(Path(source))

    def commit(self, sources: list[str]) -> None:
        """同步成功后推进账本：现存文件记新 md5，消失的从账本移除。

        注意必须在"写库确认成功"之后调用——中途失败不 commit，
        下次 list_changes() 会把同一批变更再报一遍（可重放）。
        """
        current = self._current_files()
        for source in sources:
            if source in current:
                self.state.set(source, current[source])
            else:
                self.state.remove(source)
        self.state.save()
