"""图片落盘测试（全离线）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.multimodal.image_store import ImageStore

A = b"\x89PNG\r\n\x1a\n" + b"a" * 100
B = b"\x89PNG\r\n\x1a\n" + b"b" * 100

class TestImageStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="imgstore_"))
        self.store = ImageStore(root=self.tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_save_creates_file(self):
        out = self.store.save(A)
        self.assertTrue(out.is_new)
        self.assertTrue(out.rel_path.startswith("images/"))
        self.assertTrue((self.tmp / out.rel_path).exists())
        self.assertEqual((self.tmp / out.rel_path).read_bytes(), A)

    def test_second_save_dedupes(self):
        first = self.store.save(A)
        second = self.store.save(A)
        self.assertTrue(first.is_new)
        self.assertFalse(second.is_new)
        self.assertEqual(first.rel_path, second.rel_path)
        files = list((self.tmp / "images").iterdir())
        self.assertEqual(len(files), 1, "同内容必须只留一个文件")

    def test_different_bytes_different_paths(self):
        p1 = self.store.save(A)
        p2 = self.store.save(B)
        self.assertNotEqual(p1.rel_path, p2.rel_path)

    def test_suffix_normalization(self):
        out = self.store.save(A, suffix="jpg")
        self.assertTrue(out.rel_path.endswith(".jpg"))

    def test_no_leftover_tmp(self):
        self.store.save(A)
        leftovers = [p for p in (self.tmp / "images").iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

if __name__ == "__main__":
    unittest.main()
