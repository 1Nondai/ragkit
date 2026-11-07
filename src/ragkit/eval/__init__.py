"""评测层：数据集、指标、执行器。"""

from ragkit.eval.dataset import EvalCase, load_cases, save_cases
from ragkit.eval.metrics import (
    CaseResult,
    EvalReport,
    answer_keyword_coverage,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)
from ragkit.eval.runner import compare, run_full_eval, run_retrieval_eval

__all__ = [
    "EvalCase",
    "load_cases",
    "save_cases",
    "recall_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
    "answer_keyword_coverage",
    "CaseResult",
    "EvalReport",
    "run_retrieval_eval",
    "run_full_eval",
    "compare",
]
