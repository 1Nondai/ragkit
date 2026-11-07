"""命令行：启动 HTTP 服务。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main() -> int:
    parser = argparse.ArgumentParser(description="启动 RAG HTTP 服务")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import os

    if args.memory:
        os.environ["RAG_MILVUS__URI"] = "memory://local"

    from ragkit.config import get_settings

    s = get_settings().api
    host = args.host or s.host
    port = args.port or s.port

    import uvicorn

    uvicorn.run("ragkit.api.server:app", host=host, port=port, reload=args.reload)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
