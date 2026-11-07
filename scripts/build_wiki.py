"""命令行：把语料编译成 wiki（可选增强层）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings  # noqa: E402
from ragkit.ingest.chunker import chunk_documents  # noqa: E402
from ragkit.ingest.loader import load_directory  # noqa: E402
from ragkit.wiki.compiler import compile_pages  # noqa: E402
from ragkit.wiki.manager import WikiManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="编译 wiki（chunk → 知识页）")
    parser.add_argument("--data-dir", default=None, help="语料目录，默认取配置 ingest.data_dir")
    parser.add_argument("--out", default=None, help="wiki 输出目录，默认取配置 wiki.root")
    parser.add_argument("--lint", action="store_true", help="编译后跑 lint")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.ingest.data_dir
    out_dir = Path(args.out) if args.out else settings.wiki.root

    docs = load_directory(data_dir)
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size or settings.ingest.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.ingest.chunk_overlap,
        min_chunk_chars=settings.ingest.min_chunk_chars,
    )
    pages = compile_pages(chunks)
    manager = WikiManager(out_dir)
    stats = manager.build(pages)

    print(json.dumps({"chunks": len(chunks), **stats}, ensure_ascii=False, indent=2))
    print(f"入口: {out_dir / 'index.md'}")

    if args.lint:
        findings = manager.lint()
        for name, items in findings.items():
            print(f"  {name}: {len(items)}", items[:3] if items else "")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
