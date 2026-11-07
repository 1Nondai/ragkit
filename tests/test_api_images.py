"""API 静态图路由测试（TestClient，不需要真起服务）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from ragkit.config import IngestSettings, Settings

PNG = b"\x89PNG\r\n\x1a\n" + b"p" * 200

class TestImagesRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="api_img_"))
        (cls.tmp / "images").mkdir(parents=True)
        (cls.tmp / "images" / "a.png").write_bytes(PNG)
        settings = Settings(ingest=IngestSettings(data_dir=cls.tmp / "raw"))
        import ragkit.api.server as server_mod
        import ragkit.bootstrap as bootstrap_mod

        with patch.object(server_mod, "get_settings", return_value=settings), \
             patch.object(bootstrap_mod, "get_settings", return_value=settings):
            from ragkit.api.server import create_app
            cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_existing_image_served(self):
        resp = self.client.get("/images/a.png")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, PNG)

    def test_missing_image_404(self):
        resp = self.client.get("/images/nope.png")
        self.assertEqual(resp.status_code, 404)

    def test_search_returns_image_path(self):
        from ragkit.api.schemas import Citation, Hit
        c = Citation(source="a.pdf", image_path="images/a.png")
        self.assertEqual(c.image_path, "images/a.png")
        self.assertEqual(Hit().image_path, "")

if __name__ == "__main__":
    unittest.main()
