"""统一命令行入口（`pip install -e .` 后可用 ragkit-ingest / ragkit-query / ragkit-eval）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    """开发态直接跑源码时，把 src 加入路径。"""
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

def ingest() -> int:
    """`ragkit-ingest`：入库命令（--recreate / --kb / --memory）。"""
    _bootstrap()
    from ragkit.bootstrap import build_ingest_pipeline
    from ragkit.config import get_settings

    """
    就是命令行参数怎么解析
    $ ragkit-ingest --help
    usage: ragkit-ingest [-h] [--data-dir DATA_DIR] [--recreate] [--kb KB] [--memory]

    数据入库

    options:
      --data-dir DATA_DIR
      --recreate
      --kb KB
      --memory
      -h, --help         show this help message and exit
    """
    parser = argparse.ArgumentParser(prog="ragkit-ingest", description="数据入库")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--kb", default=None, help="给这批数据打知识库标签（可按 kb 过滤检索）")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--incremental", action="store_true",
                        help="增量同步：只处理内容有变化（md5 对比）的文件，删除的文件自动清块")
    parser.add_argument("--sync-state", default=None,
                        help="增量同步状态文件路径（默认 <data_dir>/.sync_state.json）")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.ingest.data_dir
    pipeline = build_ingest_pipeline(prefer_memory=args.memory, kb=args.kb)

    if args.incremental:
        from ragkit.connectors import LocalDirConnector, SyncStateStore, incremental_sync

        state_path = Path(args.sync_state) if args.sync_state else data_dir / ".sync_state.json"
        connector = LocalDirConnector(data_dir, SyncStateStore(state_path))
        stats = incremental_sync(connector, pipeline, pipeline.store).to_dict()
    else:
        stats = pipeline.run(data_dir, recreate=args.recreate).to_dict()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0

def query() -> int:
    """`ragkit-query`：单次问答（--json 打印 trace/timings/hits，排障首选）。"""
    _bootstrap()
    from ragkit.bootstrap import build_deps
    from ragkit.graph.graph import build_graph, run

    parser = argparse.ArgumentParser(prog="ragkit-query", description="单次问答")
    parser.add_argument("question")
    parser.add_argument("--kb", default=None)
    parser.add_argument("--groups", default=None,
                        help="逗号分隔的权限组（acl 前置过滤，如 ops,finance）")
    parser.add_argument("--session", default="cli")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()

    deps = build_deps(prefer_memory=args.memory)
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    state = run(args.question, deps, session_id=args.session, kb_name=args.kb,
                groups=groups, compiled=build_graph(deps))
    if args.as_json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    else:
        print(state.get("answer", ""))
    return 0

def eval() -> int:  # noqa: A001 - 命令行子命令名就是 eval
    """`ragkit-eval`：跑评测集，产出 reports/eval_report.{md,json}。"""
    _bootstrap()
    from ragkit.bootstrap import build_deps
    from ragkit.eval.dataset import load_cases
    from ragkit.eval.runner import run_full_eval, run_retrieval_eval

    parser = argparse.ArgumentParser(prog="ragkit-eval", description="跑评测")
    parser.add_argument("--data-set", default="data/eval/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cases = load_cases(args.data_set)
    deps = build_deps(prefer_memory=args.memory)
    report = run_full_eval(deps, cases) if args.full else run_retrieval_eval(
        deps.service, cases, top_k=args.top_k
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / "eval_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.aggregate(), ensure_ascii=False, indent=2))
    return 0

def main(argv: list[str] | None = None) -> int:
    """子命令分发：`python -m ragkit.cli <ingest|query|eval> [参数]`。"""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("用法: python -m ragkit.cli <ingest|query|eval> [参数]")
        return 0
    cmd, rest = argv[0], argv[1:]
    sys.argv = [f"ragkit-{cmd}", *rest]
    handlers = {"ingest": ingest, "query": query, "eval": eval}
    if cmd not in handlers:
        print(f"未知子命令: {cmd}（可选 {sorted(handlers)}）")
        return 2
    return handlers[cmd]()

if __name__ == "__main__":
    raise SystemExit(main())
