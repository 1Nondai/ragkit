"""配置与路由测试。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings, redacted, reload_settings
from ragkit.graph.router import HybridRouter, KeywordRouter


class TestConfig(unittest.TestCase):
    def tearDown(self) -> None:
        for key in list(os.environ):
            if key.startswith("RAG_"):
                del os.environ[key]
        reload_settings()

    def test_env_nested_override(self):
        os.environ["RAG_INGEST__CHUNK_SIZE"] = "1234"
        os.environ["RAG_MILVUS__COLLECTION"] = "test_collection"
        s = reload_settings()
        self.assertEqual(s.ingest.chunk_size, 1234)
        self.assertEqual(s.milvus.collection, "test_collection")

    def test_redacted_hides_secrets(self):
        os.environ["RAG_LLM__API_KEY"] = "sk-should-not-leak"
        s = reload_settings()
        snapshot = redacted()
        self.assertTrue(s.llm.api_key)
        self.assertNotIn("sk-should-not-leak", str(snapshot))
        self.assertTrue(snapshot["llm"]["api_key_set"])

    def test_overlap_must_be_smaller_than_size(self):
        from pydantic import ValidationError

        os.environ["RAG_INGEST__CHUNK_SIZE"] = "100"
        os.environ["RAG_INGEST__CHUNK_OVERLAP"] = "100"
        with self.assertRaises(ValidationError):
            reload_settings()

    def test_settings_is_singleton_until_reload(self):
        a = get_settings()
        b = get_settings()
        self.assertIs(a, b)
        reload_settings()
        self.assertIsNot(a, get_settings())

class TestRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = HybridRouter(KeywordRouter(), llm_router=None)

    def test_greeting_is_chitchat(self):
        self.assertEqual(self.router.decide("你好").route, "chitchat")

    def test_vague_question_needs_clarify(self):
        self.assertEqual(self.router.decide("这个是怎么办？").route, "clarify")

    def test_trouble_keyword_routes_to_trouble_kb(self):
        decision = self.router.decide("服务启动报错，如何排查？")
        self.assertEqual(decision.route, "kb")
        self.assertEqual(decision.kb_name, "trouble")

    def test_default_is_kb(self):
        self.assertEqual(self.router.decide("Milvus 是什么").route, "kb")

    def test_rule_hit_skips_llm(self):
        class Boom:
            def decide(self, question, history=None):
                raise AssertionError("规则命中时不应调用 LLM")

        r = HybridRouter(KeywordRouter(), Boom())  # type: ignore[arg-type]
        self.assertEqual(r.decide("你好").matched_rule, "chitchat")

if __name__ == "__main__":
    unittest.main()
