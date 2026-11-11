"""把评测集导出为 Langfuse dataset，并以实验形式跑分（recall/hit_rate/mrr/answer_coverage 挂到每条 trace）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragkit.eval.dataset import EvalCase
from ragkit.eval.metrics import (
    answer_keyword_coverage,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)
from ragkit.graph.nodes import Deps
from ragkit.logging_setup import get_logger

log = get_logger("eval.langfuse_export")

DEFAULT_DATASET = "ragkit-eval"

def _require_client(sink: Any) -> Any:
    """从 Deps 里拿 Langfuse 客户端；观测未启用/不是真实现时报错退出。"""
    if sink is None or not getattr(sink, "active", False) or not hasattr(sink, "_client"):
        raise RuntimeError(
            "Langfuse 未启用（RAG_OBS__ENABLED=false 或缺少 key）；"
            "评测导出需要先启用观测层"
        )
    return sink._client

def export_dataset(
    client: Any,
    cases: Sequence[EvalCase],
    *,
    name: str = DEFAULT_DATASET,
) -> int:
    """把评测用例写成 Langfuse dataset，返回写入条数。重复执行会追加重复条目（先删旧 dataset 再导）。"""
    try:
        client.create_dataset(
            name=name,
            description="ragkit 评测集：fact/paraphrase/multi_hop/numeric/cross_doc/negative 六类",
        )
        log.info("已创建 Langfuse dataset: %s", name)
    except Exception as exc:  # noqa: BLE001 - 已存在等情况下继续写入条目
        log.info("dataset %s 可能已存在（%s），继续写入条目", name, exc)

    written = 0
    for case in cases:
        client.create_dataset_item(
            dataset_name=name,
            input={"question": case.question, "kb": case.kb},
            expected_output={
                "keywords": case.expected_keywords,
                "reference_answer": case.reference_answer,
            },
            metadata={"category": case.category, "image_gold": case.image_gold,
                      "note": case.note},
        )
        written += 1
    log.info("已写入 %d 条评测条目到 dataset %s", written, name)
    return written

def run_eval_experiment(
    deps: Deps,
    cases: Sequence[EvalCase],
    *,
    run_name: str,
    dataset_name: str = DEFAULT_DATASET,
    max_concurrency: int = 4,
) -> Any:
    """以 Langfuse 实验跑一遍全链路评测，返回 ExperimentResult（含 UI 链接）。

    每条用例 = 一次完整 graph.run()（带 trace），evaluator 分数挂在对应 trace 上。
    并发默认压到 4：全链路要打真模型，无脑拉满并发只会触发限流。
    """
    client = _require_client(getattr(deps, "trace_sink", None))
    from langfuse.experiment import Evaluation

    data = [
        {
            "input": {"question": case.question, "kb": case.kb},
            "expected_output": {
                "keywords": case.expected_keywords,
                "reference_answer": case.reference_answer,
            },
            "metadata": {"category": case.category},
        }
        for case in cases
    ]

    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        """跑一次完整问答，产出 evaluator 需要的字段。"""
        from ragkit.graph.graph import run

        q = (item.get("input") or {}).get("question") or ""
        kb = (item.get("input") or {}).get("kb")
        state = run(q, deps, kb_name=kb)
        return {
            "answer": state.get("answer", ""),
            "hit_texts": [h.get("text", "") for h in state.get("hits", [])],
            "grounded": state.get("grounded", False),
            "confidence": state.get("confidence", 0.0),
            "timings_ms": state.get("timings", {}),
            "route": state.get("route", ""),
        }

    def evaluators(
        *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
    ) -> list[Any]:
        """检索质量三指标 + 生成覆盖度；keywords 为空的题（negative 类）跳过检索指标。"""
        kws = (expected_output or {}).get("keywords") or []
        texts = (output or {}).get("hit_texts") or []
        scores: list[Any] = []
        if kws:
            scores = [
                Evaluation(name="recall", value=round(recall_at_k(kws, texts), 4)),
                Evaluation(name="hit_rate", value=round(hit_rate_at_k(kws, texts), 4)),
                Evaluation(name="mrr", value=round(mrr_at_k(kws, texts), 4)),
            ]
        reference = (expected_output or {}).get("reference_answer") or ""
        answer = (output or {}).get("answer") or ""
        if reference:
            scores.append(
                Evaluation(
                    name="answer_coverage",
                    value=round(answer_keyword_coverage(reference, answer), 4),
                )
            )
        return scores

    result = client.run_experiment(
        name=run_name,
        data=data,
        task=task,
        evaluators=[evaluators],
        max_concurrency=max_concurrency,
        metadata={"dataset": dataset_name, "size": len(cases)},
    )
    return result
