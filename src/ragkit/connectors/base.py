"""连接器抽象：数据源 → 变更清单 → 文档。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ragkit.ingest.loader import LoadedDocument


class ChangeType(str, Enum):
    """一条变更的类型。"""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"

@dataclass
class ChangeRecord:
    """一条变更记录。"""

    source: str
    filename: str
    change: ChangeType

@runtime_checkable
class SourceConnector(Protocol):
    """数据源连接器最小接口。"""

    name: str

    def list_changes(self) -> list[ChangeRecord]: ...

    def fetch(self, source: str) -> LoadedDocument: ...

    def commit(self, sources: list[str]) -> None: ...
