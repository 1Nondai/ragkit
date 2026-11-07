"""一键离线演示：入库 → 检索 → 生成，全程不需要 Milvus、不需要模型 Key。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_deps, build_ingest_pipeline  # noqa: E402
from ragkit.config import get_settings  # noqa: E402
from ragkit.graph.graph import build_graph, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser(description="离线一键演示")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--question", default="Milvus 是什么，它解决什么问题？")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.ingest.data_dir

    print(f"① 入库: {data_dir}")
    pipeline = build_ingest_pipeline(prefer_memory=True)
    print("   ", pipeline.run(data_dir, recreate=True).to_dict())

    print("② 组装依赖（内存库 + 离线模型）")
    deps = build_deps(prefer_memory=True)
    app = build_graph(deps)
    print("    图类型:", type(app).__name__)

    print(f"③ 提问: {args.question}")
    state = run(args.question, deps, session_id="demo", compiled=app)

    print("\n----- 执行轨迹 -----")
    print(" → ".join(state.get("trace", [])))
    print("耗时(ms):", state.get("timings", {}))

    print("\n----- 检索到的片段 -----")
    for i, h in enumerate(state.get("hits", [])[:3], start=1):
        print(f"[{i}] {h.get('source')} | {h.get('title')} | score={h.get('score')}")
        print("   ", (h.get("text") or "")[:160].replace("\n", " "))

    print("\n----- 答案 -----")
    print(state.get("answer", ""))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
