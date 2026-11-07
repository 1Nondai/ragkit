"""接入层测试：变更检测 + 增量同步（全离线，内存库 + 哈希向量）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.connectors import (
    ChangeType,
    LocalDirConnector,
    SyncStateStore,
    incremental_sync,
)
from ragkit.embeddings import HashingEmbedding
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.retrieval.protocol import InMemoryStore


def _make_pipeline(store: InMemoryStore) -> IngestPipeline:
    return IngestPipeline(store=store, embeddings=HashingEmbedding(dim=32))

class TestSyncStateStore(unittest.TestCase):
    def test_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            store = SyncStateStore(path)
            store.set("a.md", "m1")
            store.save()
            store2 = SyncStateStore(path)
            self.assertEqual(store2.get("a.md"), {"md5": "m1"})

    def test_corrupted_file_falls_back_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{broken json", encoding="utf-8")
            store = SyncStateStore(path)
            self.assertIsNone(store.get("a.md"))

class TestLocalDirConnector(unittest.TestCase):
    def _connector(self, root: Path) -> LocalDirConnector:
        return LocalDirConnector(root, SyncStateStore(root / ".sync_state.json"))

    def test_added_modified_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\n内容一", encoding="utf-8")
            conn = self._connector(root)

            changes = {c.source: c.change for c in conn.list_changes()}
            self.assertEqual(len(changes), 1)
            self.assertEqual(list(changes.values())[0], ChangeType.ADDED)

            conn.commit(list(changes))

            (root / "a.md").write_text("# A\n\n内容二（改了）", encoding="utf-8")
            (root / "b.md").write_text("# B\n\n新文件", encoding="utf-8")
            changes = {c.filename: c.change for c in conn.list_changes()}
            self.assertEqual(changes["a.md"], ChangeType.MODIFIED)
            self.assertEqual(changes["b.md"], ChangeType.ADDED)
            conn.commit([c.source for c in conn.list_changes()])

            (root / "b.md").unlink()
            changes = {c.filename: c.change for c in conn.list_changes()}
            self.assertEqual(changes["b.md"], ChangeType.DELETED)

    def test_no_changes_after_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\n内容", encoding="utf-8")
            conn = self._connector(root)
            conn.commit([c.source for c in conn.list_changes()])
            self.assertEqual(conn.list_changes(), [])

    def test_unsupported_suffix_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A", encoding="utf-8")
            (root / "data.png").write_bytes(b"\x89PNG")
            conn = self._connector(root)
            self.assertEqual(len(conn.list_changes()), 1)

class TestIncrementalSync(unittest.TestCase):
    def _setup(self, root: Path):
        store = InMemoryStore()
        pipeline = _make_pipeline(store)
        conn = LocalDirConnector(root, SyncStateStore(root / ".sync_state.json"))
        return store, pipeline, conn

    def test_added_modified_deleted_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, pipeline, conn = self._setup(root)

            (root / "a.md").write_text(
                "# 文档甲\n\n" + "Alpha 内容。" * 20, encoding="utf-8")
            (root / "b.md").write_text(
                "# 文档乙\n\n" + "Beta 内容。" * 20, encoding="utf-8")
            pipeline.run(root)
            total_full = store.count()
            self.assertGreater(total_full, 0)

            stats = incremental_sync(conn, pipeline, store)
            self.assertEqual((stats.added, stats.modified, stats.deleted), (2, 0, 0))
            self.assertEqual(store.count(), total_full)

            stats = incremental_sync(conn, pipeline, store)
            self.assertEqual((stats.added, stats.modified, stats.deleted), (0, 0, 0))
            self.assertEqual(store.count(), total_full)

            (root / "a.md").write_text(
                "# 文档甲改\n\n" + "Alpha 内容已更新。" * 20, encoding="utf-8")
            (root / "b.md").unlink()
            stats = incremental_sync(conn, pipeline, store)
            self.assertEqual((stats.added, stats.modified), (0, 1))
            self.assertEqual(stats.deleted, 1)
            self.assertGreater(stats.chunks_deleted, 0)

            leftovers = [
                c for c in store._chunks.values()
                if c.metadata.get("filename") == "b.md"
            ]
            self.assertEqual(leftovers, [])
            a_titles = {
                str(c.metadata.get("title", ""))
                for c in store._chunks.values() if c.metadata.get("filename") == "a.md"
            }
            self.assertNotIn("文档甲", a_titles)
            self.assertTrue(any("文档甲改" in t for t in a_titles))

    def test_failed_fetch_does_not_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, pipeline, conn = self._setup(root)
            (root / "a.md").write_text("# A\n\n内容一。" * 10, encoding="utf-8")
            pipeline.run(root)
            conn.commit([c.source for c in conn.list_changes()])

            (root / "a.md").write_text("# A\n\n内容二。" * 10, encoding="utf-8")
            (root / "bad.md").write_text("# Bad\n\n坏文件内容。" * 10, encoding="utf-8")
            conn.fetch = lambda source: (_ for _ in ()).throw(
                RuntimeError("boom")) if source.endswith("bad.md") else (
                LocalDirConnector.fetch(conn, source))
            stats = incremental_sync(conn, pipeline, store)
            self.assertIn(str((root / "bad.md").resolve()), stats.failed)
            pending = {c.filename for c in conn.list_changes()}
            self.assertIn("bad.md", pending)

if __name__ == "__main__":
    unittest.main()
