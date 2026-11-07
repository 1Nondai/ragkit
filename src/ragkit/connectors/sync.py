"""增量同步：连接器报变更 → 同步层决定删什么、嵌什么 → 成功后推进账本。

删除必须先于写入处理：文件被"移动/改名"时会同时出现 DELETED(旧路径) 与
ADDED(新路径)，先删后写能保证旧块清干净、新块正常进。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragkit.connectors.base import ChangeRecord, ChangeType, SourceConnector
from ragkit.ingest.chunker import Chunk
from ragkit.logging_setup import get_logger
from ragkit.retrieval.protocol import VectorStore

log = get_logger("connectors.sync")

@dataclass
class SyncStats:
    """一次增量同步的统计（对齐 IngestStats 的可观测口径）。"""

    added: int = 0
    modified: int = 0
    deleted: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    failed: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "failed": self.failed,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

def _delete_source_chunks(store: VectorStore, source: str) -> int:
    """删掉某 source 的全部旧块。库不支持删除时抛错——此时增量不可用，让上层决定回退全量。"""
    return store.delete_by_source(source)

def incremental_sync(
    connector: SourceConnector,
    pipeline,
    store: VectorStore,
    *,
    force_sources: Sequence[str] = (),
) -> SyncStats:
    """跑一次增量同步。"""
    started = time.perf_counter()
    stats = SyncStats()

    changes = list(connector.list_changes())
    if force_sources:
        known = {c.source for c in changes if c.change != ChangeType.DELETED}
        changes = [c for c in changes if c.source not in known]
        changes += [
            ChangeRecord(source=s, filename=s.replace("\\", "/").rsplit("/", 1)[-1],
                         change=ChangeType.MODIFIED)
            for s in force_sources if s not in known
        ]

    deletes = [c for c in changes if c.change == ChangeType.DELETED]
    upserts = [c for c in changes if c.change != ChangeType.DELETED]

    for rec in deletes:
        stats.chunks_deleted += _delete_source_chunks(store, rec.source)
        stats.deleted += 1
        log.info("已删除文档块: %s", rec.source)

    for rec in upserts:
        try:
            doc = connector.fetch(rec.source)
            chunks: list[Chunk] = pipeline.chunk([doc])
            if not chunks:
                log.warning("文档没有产出任何块，跳过: %s", rec.source)
                stats.failed.append(rec.source)
                continue
            stats.chunks_deleted += _delete_source_chunks(store, rec.source)
            vectors = pipeline.embed(chunks)
            image_vectors, _n = pipeline.embed_images(chunks)
            stats.chunks_written += pipeline.write(chunks, vectors, image_vectors)
            stats.added += rec.change == ChangeType.ADDED
            stats.modified += rec.change == ChangeType.MODIFIED
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不拖垮整批
            log.warning("增量同步失败，跳过 %s: %s", rec.source, exc)
            stats.failed.append(rec.source)

    ok = [c.source for c in changes
          if c.change == ChangeType.DELETED or c.source not in stats.failed]
    if ok:
        connector.commit(ok)

    stats.elapsed_ms = (time.perf_counter() - started) * 1000
    log.info("增量同步完成: %s", stats.to_dict())
    return stats
