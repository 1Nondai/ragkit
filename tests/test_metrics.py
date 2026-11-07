"""
指标测试：口径必须对。

指标写错比没有指标更危险——它会让错误的优化看起来像进步。
这里把 recall / hit / mrr / 覆盖率 的边界情况全部钉死。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.eval.metrics import (
    answer_keyword_coverage,
    hit_rate_at_k,
    mrr_at_k,
    percentile,
    recall_at_k,
)


class TestMetrics(unittest.TestCase):
    def test_recall_full_and_partial(self):
        docs = ["Milvus 支持 HNSW 与 IVF_FLAT 索引", "数据存在 MinIO"]
        self.assertAlmostEqual(recall_at_k(["HNSW", "MinIO"], docs), 1.0)
        self.assertAlmostEqual(recall_at_k(["HNSW", "DiskANN"], docs), 0.5)
        self.assertAlmostEqual(recall_at_k(["不存在"], docs), 0.0)

    def test_recall_is_case_insensitive(self):
        self.assertAlmostEqual(recall_at_k(["hnsw"], ["Milvus HNSW"]), 1.0)

    def test_recall_without_keywords_is_zero(self):
        self.assertEqual(recall_at_k([], ["任意文本"]), 0.0)

    def test_hit_rate_is_binary(self):
        docs = ["a", "b"]
        self.assertEqual(hit_rate_at_k(["x", "a"], docs), 1.0)
        self.assertEqual(hit_rate_at_k(["x", "y"], docs), 0.0)

    def test_mrr_reflects_position(self):
        docs = ["无关内容", "无关内容", "这里有关键词 HNSW"]
        self.assertAlmostEqual(mrr_at_k(["HNSW"], docs), 1 / 3)
        self.assertAlmostEqual(mrr_at_k(["HNSW"], ["HNSW 在第一条"]), 1.0)
        self.assertEqual(mrr_at_k(["不存在"], docs), 0.0)

    def test_coverage_between_zero_and_one(self):
        ref = "Milvus 使用 HNSW 索引，数据存于 MinIO"
        self.assertGreater(answer_keyword_coverage(ref, ref), 0.9)
        self.assertLess(answer_keyword_coverage(ref, "完全无关的一句话"), 0.3)

    def test_percentile(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, 0.5), 3)
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertEqual(percentile([], 0.5), 0.0)

if __name__ == "__main__":
    unittest.main()
