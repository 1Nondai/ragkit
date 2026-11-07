"""命令行：离线一键评测（入库 → 评测 → 出报告）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_deps, build_ingest_pipeline  # noqa: E402
from ragkit.eval.dataset import load_cases  # noqa: E402
from ragkit.eval.runner import compare, run_retrieval_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser(description="离线检索评测（内存库，一键可复现）")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--data-set", default=str(ROOT / "data" / "eval" / "eval_set.jsonl"))
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 retrieval.top_k")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 ingest.chunk_size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="覆盖 ingest.chunk_overlap")
    parser.add_argument("--a-b", action="store_true", help="额外跑 top_k=5 vs 20 的逐题 diff")
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    from ragkit.config import get_settings

    settings = get_settings()
    if args.chunk_size:
        settings.ingest.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.ingest.chunk_overlap = args.chunk_overlap

    cases = load_cases(args.data_set)
    if not cases:
        print(f"评测集为空: {args.data_set}")
        return 1

    pipeline = build_ingest_pipeline(prefer_memory=True)
    stats = pipeline.run(Path(args.data_dir), recreate=True)
    print("=== 入库 ===")
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))

    deps = build_deps(prefer_memory=True)
    report = run_retrieval_eval(deps.service, cases, top_k=args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (out_dir / "eval_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 评测汇总 ===")
    for key, value in report.aggregate().items():
        print(f"  {key}: {value}")
    print(f"\n报告已写入: {out_dir / 'eval_report.md'}")

    if args.a_b:
        before = run_retrieval_eval(deps.service, cases, top_k=5)
        after = run_retrieval_eval(deps.service, cases, top_k=20)
        print("\n=== A/B 逐题 diff (top_k 5 → 20) ===")
        for row in compare(before, after):
            print(f"  {row['question'][:28]:30s} recall {row['recall_before']:.2f} → "
                  f"{row['recall_after']:.2f}  Δ={row['delta']:+.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
