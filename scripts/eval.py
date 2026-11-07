"""命令行：跑评测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_deps  # noqa: E402
from ragkit.eval.dataset import load_cases  # noqa: E402
from ragkit.eval.runner import run_full_eval, run_retrieval_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--data-set", default=str(ROOT / "data" / "eval" / "eval_set.jsonl"))
    parser.add_argument("--data-dir", default=None, help="内存模式下先从这个目录入库")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="连生成一起评（需配置 LLM）")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    cases = load_cases(args.data_set)
    if not cases:
        print(f"评测集为空: {args.data_set}")
        return 1

    deps = build_deps(prefer_memory=args.memory)

    if args.memory and deps.service.store.count() == 0:
        from pathlib import Path as _Path

        from ragkit.bootstrap import build_ingest_pipeline
        from ragkit.config import get_settings as _gs

        data_dir = _Path(args.data_dir) if args.data_dir else _gs().ingest.data_dir
        print(f"内存库为空，先入库: {data_dir}")
        build_ingest_pipeline(prefer_memory=True).run(_Path(data_dir), recreate=True)

    report = run_full_eval(deps, cases) if args.full else run_retrieval_eval(
        deps.service, cases, top_k=args.top_k
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (out_dir / "eval_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = report.aggregate()
    print("\n=== 评测汇总 ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\n报告已写入: {out_dir / 'eval_report.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
