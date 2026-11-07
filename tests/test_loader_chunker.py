"""加载与切片测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.ingest.chunker import chunk_documents, chunk_sections, make_chunk_id, recursive_split
from ragkit.ingest.loader import LoadedDocument, Section, load_directory, load_file, parse_markdown

MD = """# 产品手册

## 安装

第一步：准备环境。
第二步：执行安装命令。

```python
print("代码块应被丢弃")
```

## 故障排查

### 错误码 E01

重启服务即可恢复。这里的句子用来验证递归切分是否会把长段落切开并保持重叠。
""" * 3

class TestLoader(unittest.TestCase):
    def test_parse_markdown_builds_title_path(self):
        sections = parse_markdown(MD)
        titles = [s.title for s in sections]
        self.assertIn("产品手册 -> 安装", titles)
        self.assertIn("产品手册 -> 故障排查 -> 错误码 E01", titles)

    def test_code_block_is_dropped(self):
        body = "\n".join(s.text for s in parse_markdown(MD))
        self.assertNotIn("代码块应被丢弃", body)

    def test_load_file_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.md"
            path.write_text(MD, encoding="utf-8")
            doc = load_file(path)
            self.assertEqual(doc.filename, "demo.md")
            self.assertGreater(doc.section_count, 0)
            self.assertTrue(all(s.metadata["filename"] == "demo.md" for s in doc.sections))

    def test_load_directory_is_sorted_and_resilient(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text("# A\n\n内容", encoding="utf-8")
            (Path(tmp) / "b.txt").write_text("纯文本", encoding="utf-8")
            (Path(tmp) / "c.pdf").write_bytes(b"%PDF-1.4 not really")
            docs = load_directory(Path(tmp))
            self.assertEqual([d.filename for d in docs], ["a.md", "b.txt"])

class TestChunker(unittest.TestCase):
    def _doc(self) -> LoadedDocument:
        return LoadedDocument(
            source="/tmp/x.md", filename="x.md", filetype="md",
            sections=[Section(text="这是一个用于测试的段落。" * 40, title_path=["A", "B"], level=2)],
        )

    def test_recursive_split_respects_size(self):
        pieces = recursive_split("这是一个很长的段落，需要被切开。" * 50, size=200, overlap=40)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 260 for p in pieces), "切分后不应出现远大于 size 的块")

    def test_chunk_id_is_stable_and_sensitive(self):
        a = make_chunk_id("同一段文本", "标题", "a.md")
        b = make_chunk_id("同一段文本", "标题", "a.md")
        c = make_chunk_id("同一段文本", "标题", "b.md")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_min_chars_filters_noise(self):
        doc = LoadedDocument(source="/tmp/y.md", filename="y.md", filetype="md",
                             sections=[Section(text="短", title_path=["A"])])
        self.assertEqual(chunk_sections(doc, min_chunk_chars=30), [])

    def test_chunk_documents_dedupes(self):
        doc = self._doc()
        chunks = chunk_documents([doc, doc], chunk_size=300, chunk_overlap=50)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "重复文档不应产生重复 chunk")

    def test_metadata_completeness(self):
        chunks = chunk_sections(self._doc(), chunk_size=300, chunk_overlap=50)
        required = {"chunk_id", "doc_id", "source", "filename", "title", "category"}
        self.assertTrue(required.issubset(chunks[0].metadata.keys()))

if __name__ == "__main__":
    unittest.main()
