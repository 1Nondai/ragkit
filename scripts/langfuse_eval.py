"""Langfuse 评测入口：导出 dataset + 跑可对比实验（前置：RAG_OBS__ENABLED=true 且已配 key）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.bootstrap import build_deps
from ragkit.config import get_settings
from ragkit.eval.dataset import load_cases
from ragkit.eval.langfuse_export import DEFAULT_DATASET, export_dataset, run_eval_experiment
from ragkit.logging_setup import get_logger

log = get_logger("scripts.langfuse_eval")

def main() -> int:
    parser = argparse.ArgumentParser(description="Langfuse 评测导出与实验")
    parser.add_argument("--data", default="data/eval/eval_set.jsonl", help="评测集路径")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Langfuse dataset 名")
    parser.add_argument("--run-name", default=f"ragkit-{get_settings().obs.environment}",
                        help="实验名：同一 run_name 的多轮在 UI 里自动归到同一实验下对比")
    parser.add_argument("--concurrency", type=int, default=4, help="实验并发（默认 4）")
    parser.add_argument("--export-only", action="store_true", help="只导数据集，不跑实验")
    parser.add_argument("--run-only", action="store_true", help="跳过导数据集，直接跑实验")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.obs.enabled:
        log.error("Langfuse 未启用：请先设置 RAG_OBS__ENABLED=true 与 RAG_OBS__PUBLIC_KEY/SECRET_KEY")
        return 1

    deps = build_deps(prefer_memory=settings.milvus.uri.startswith("memory"))
    sink = getattr(deps, "trace_sink", None)
    if sink is None or not getattr(sink, "active", False) or not hasattr(sink, "_client"):
        log.error("观测汇未激活（SDK 缺失或初始化失败），无法导出")
        return 1

    cases = load_cases(args.data)
    if not cases:
        log.error("评测集为空: %s", args.data)
        return 1
    log.info("加载评测集 %d 条（%s）", len(cases), args.data)

    client = sink._client
    if not args.run_only:
        export_dataset(client, cases, name=args.dataset)

    if args.export_only:
        return 0

    result = run_eval_experiment(
        deps, cases, run_name=args.run_name, dataset_name=args.dataset,
        max_concurrency=args.concurrency,
    )
    url = getattr(result, "experiment_url", None)
    log.info("实验完成：%s", url or "(未取到 UI 链接)")
    for item in getattr(result, "item_results", []):
        log.info("  %s -> %s", item.item.get("input", {}).get("question", ""),
                 {e.name: e.value for e in item.evaluations})
    if url:
        print(url)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
