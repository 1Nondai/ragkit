"""评测执行器：跑一遍评测集，产出可对比的报告。"""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.eval.dataset import EvalCase
from ragkit.eval.metrics import (
    CaseResult,
    EvalReport,
    answer_keyword_coverage,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)
from ragkit.graph.nodes import Deps
from ragkit.logging_setup import get_logger
from ragkit.retrieval.service import RetrievalService

log = get_logger("eval.runner")

def run_retrieval_eval(
    service: RetrievalService,
    cases: Sequence[EvalCase],
    *,
    top_k: int | None = None,
) -> EvalReport:
    """只评检索：不调模型，离线可跑。"""
    from ragkit.eval.metrics import image_hit_rate

    report = EvalReport()
    gold_flags: list[bool] = []
    hits_lists: list[list[dict]] = []
    for case in cases:
        ctx = service.search(case.question, top_k=top_k)
        texts = [h.text for h in ctx.hits]
        missing = [
            kw for kw in case.expected_keywords
            if not any(kw.lower() in t.lower() for t in texts)
        ]
        hit = [kw for kw in case.expected_keywords if kw not in missing]
        gold_flags.append(case.image_gold)
        hits_lists.append([{"modality": h.modality} for h in ctx.hits])

        report.cases.append(
            CaseResult(
                question=case.question,
                category=case.category,
                recall=recall_at_k(case.expected_keywords, texts),
                hit=hit_rate_at_k(case.expected_keywords, texts),
                mrr=mrr_at_k(case.expected_keywords, texts),
                latency_ms=ctx.elapsed_ms,
                hit_keywords=hit,
                missed_keywords=missing,
            )
        )
    img_rate = image_hit_rate(gold_flags, hits_lists)
    if any(gold_flags):
        log.info("图片命中率: %.2f (%s)", img_rate,
                 f"{sum(gold_flags)} 道图 gold 题")
    log.info("检索评测完成: %s", report.aggregate())
    return report

def run_full_eval(deps: Deps, cases: Sequence[EvalCase]) -> EvalReport:
    """检索 + 生成一起评（需要图依赖）。"""
    from ragkit.graph.graph import run

    report = EvalReport()
    for case in cases:
        state = run(case.question, deps, kb_name=case.kb)
        hits = state.get("hits", [])
        texts = [h.get("text", "") for h in hits]
        answer = state.get("answer", "")
        missing = [
            kw for kw in case.expected_keywords
            if not any(kw.lower() in t.lower() for t in texts)
        ]
        report.cases.append(
            CaseResult(
                question=case.question,
                category=case.category,
                recall=recall_at_k(case.expected_keywords, texts),
                hit=hit_rate_at_k(case.expected_keywords, texts),
                mrr=mrr_at_k(case.expected_keywords, texts),
                coverage=answer_keyword_coverage(case.reference_answer, answer),
                latency_ms=float(state.get("retrieval_elapsed_ms", 0.0)),
                hit_keywords=[kw for kw in case.expected_keywords if kw not in missing],
                missed_keywords=missing,
                answer=answer,
                route=state.get("route", ""),
            )
        )
    log.info("全流程评测完成: %s", report.aggregate())
    return report

def compare(before: EvalReport, after: EvalReport) -> list[dict]:
    rows = []
    for b, a in zip(before.cases, after.cases):
        rows.append(
            {
                "question": b.question,
                "recall_before": round(b.recall, 3),
                "recall_after": round(a.recall, 3),
                "delta": round(a.recall - b.recall, 3),
                "mrr_before": round(b.mrr, 3),
                "mrr_after": round(a.mrr, 3),
            }
        )
    return rows
