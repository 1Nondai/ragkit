"""命令行：量一下 wiki 层到底值不值（命中率 + 命中页质量）。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings  # noqa: E402
from ragkit.eval.dataset import load_cases  # noqa: E402
from ragkit.wiki.manager import WikiManager  # noqa: E402
from ragkit.wiki.router import WikiRouter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser(description="wiki 命中率与质量统计")
    parser.add_argument("--wiki", default=None)
    parser.add_argument("--data-set", default=str(ROOT / "data" / "eval" / "eval_set.jsonl"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None,
                        help="内容词覆盖率门槛（默认取配置 wiki.min_coverage=0.2）")
    args = parser.parse_args()

    settings = get_settings()
    root = Path(args.wiki) if args.wiki else settings.wiki.root
    threshold = args.threshold if args.threshold is not None else settings.wiki.hit_threshold

    manager = WikiManager(root)
    pages = manager.load()
    if not pages:
        print(f"wiki 为空: {root}（先跑 python scripts/build_wiki.py）")
        return 1

    router = WikiRouter(pages, threshold=threshold, stale_days=settings.wiki.stale_days,
                        min_coverage=(settings.wiki.min_coverage
                                      if args.min_coverage is None else args.min_coverage))
    cases = load_cases(args.data_set)

    decisions = [(case, router.decide(case.question)) for case in cases]
    hits = [(case, d) for case, d in decisions if d.hit]
    gates = Counter(d.gate for _, d in decisions if not d.hit)

    coverages = []
    for case, decision in hits:
        if decision.page is None or not case.expected_keywords:
            continue
        body = decision.page.body.lower()
        coverages.append(
            sum(1 for kw in case.expected_keywords if kw.lower() in body) / len(case.expected_keywords)
        )

    stats = manager.stats()
    stale = sum(1 for p in pages if p.is_stale(date.today().isoformat()))
    with_sources = sum(1 for p in pages if p.sources)
    reuse = sum(1 for _, n in (stats.get("by_slug") or {}).items() if int(n) >= 2)

    report = {
        "pages": len(pages),
        "threshold": threshold,
        "queries": len(cases),
        "hit_rate": round(len(hits) / len(cases), 4) if cases else 0.0,
        "blocked_by_gate": dict(gates),
        "hit_keyword_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "stale_rate": round(stale / len(pages), 4),
        "source_rate": round(with_sources / len(pages), 4),
        "reuse_rate": round(reuse / max(1, len(stats.get("by_slug") or {})), 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n逐条：")
    for case, decision in decisions:
        mark = "HIT " if decision.hit else "miss"
        print(f"  {mark} score={decision.score:.2f} gate={decision.gate or '-':10s} {case.question[:40]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
