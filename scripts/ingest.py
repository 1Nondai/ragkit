"""命令行：数据入库。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_ingest_pipeline  # noqa: E402
from ragkit.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 数据入库")
    parser.add_argument("--data-dir", default=None, help="文档目录（默认取配置里的 ingest.data_dir）")
    parser.add_argument("--recreate", action="store_true", help="删除并重建集合")
    parser.add_argument("--memory", action="store_true", help="使用内存向量库（不落盘）")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.ingest.data_dir

    pipeline = build_ingest_pipeline(prefer_memory=args.memory)
    stats = pipeline.run(Path(data_dir), recreate=args.recreate)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
