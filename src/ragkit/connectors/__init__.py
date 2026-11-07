"""数据接入层：连接器（变更检测）+ 同步状态 + 增量入库。"""

from ragkit.connectors.base import ChangeRecord, ChangeType, SourceConnector
from ragkit.connectors.local import LocalDirConnector
from ragkit.connectors.state import SyncStateStore
from ragkit.connectors.sync import incremental_sync

__all__ = [
    "ChangeRecord",
    "ChangeType",
    "LocalDirConnector",
    "SourceConnector",
    "SyncStateStore",
    "incremental_sync",
]
