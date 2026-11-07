"""评测集质量测试：结构合法、问题唯一、关键词都能在语料里找到。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VALID_CATEGORIES = {"fact", "paraphrase", "multi_hop", "numeric", "cross_doc", "negative"}
EVAL_PATH = Path(__file__).resolve().parents[1] / "data/eval/eval_set.jsonl"
RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw"

class TestEvalSetQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = [json.loads(line) for line in
                     EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        cls.corpus = {p.name: p.read_text(encoding="utf-8", errors="ignore").lower()
                      for p in RAW_DIR.glob("*.md")}

    def test_enough_cases(self) -> None:
        self.assertGreaterEqual(len(self.cases), 30)

    def test_categories_covered(self) -> None:
        cats = {c.get("category") for c in self.cases}
        self.assertTrue(VALID_CATEGORIES >= cats, cats)
        for must in {"fact", "paraphrase", "multi_hop", "numeric", "negative"}:
            self.assertIn(must, cats)

    def test_questions_unique(self) -> None:
        questions = [c["question"] for c in self.cases]
        self.assertEqual(len(questions), len(set(questions)))

    def test_keywords_in_corpus(self) -> None:
        misses = []
        for c in self.cases:
            if c.get("category") == "negative":
                self.assertEqual(c["expected_keywords"], [],
                                 f"negative 用例不应有期望关键词: {c['question']}")
                continue
            for kw in c["expected_keywords"]:
                if not any(kw.lower() in text for text in self.corpus.values()):
                    misses.append(f"{c['question']} -> {kw}")
        self.assertEqual(misses, [])

    def test_schema_fields(self) -> None:
        for c in self.cases:
            self.assertIn("question", c)
            self.assertIn("expected_keywords", c)
            self.assertIsInstance(c["expected_keywords"], list)

if __name__ == "__main__":
    unittest.main()
