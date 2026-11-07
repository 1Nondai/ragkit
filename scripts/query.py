"""命令行：单次问答（调试用）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_deps  # noqa: E402
from ragkit.graph.graph import build_graph, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 单次问答")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--kb", default=None, help="指定知识库名")
    parser.add_argument("--session", default="cli", help="会话 ID（多轮）")
    parser.add_argument("--memory", action="store_true", help="使用内存向量库")
    parser.add_argument("--json", action="store_true", help="输出完整状态 JSON")
    args = parser.parse_args()

    deps = build_deps(prefer_memory=args.memory)
    app = build_graph(deps)
    state = run(args.question, deps, session_id=args.session, kb_name=args.kb, compiled=app)

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return 0

    print("\n【问题】", args.question)
    print("【路由】", state.get("route", "-"), "｜ 轨迹:", " → ".join(state.get("trace", [])))
    print("【检索】", len(state.get("hits", [])), "条命中，耗时",
          state.get("timings", {}).get("retrieve_total", state.get("retrieval_elapsed_ms", 0)), "ms")
    for h in state.get("hits", [])[:5]:
        print(f"   - {h.get('source')} | {h.get('title')} | score={h.get('score')}")
    print("\n【答案】\n", state.get("answer", ""))
    print("\n【引用】")
    for c in state.get("citations", [])[:5]:
        print(f"   - {c['source']} {c['title']} (score={c['score']})")
    print("\n【耗时(ms)】", json.dumps(state.get("timings", {}), ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
