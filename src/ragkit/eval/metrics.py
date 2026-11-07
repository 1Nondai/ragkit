"""评测指标：只用**可解释、可复算**的指标，不搞"AI 打分"。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    return (text or "").lower()

def _contains(text: str, keyword: str) -> bool:
    return _normalize(keyword) in _normalize(text)

def recall_at_k(keywords: Sequence[str], retrieved_texts: Sequence[str]) -> float:
    """关键词召回率：命中关键词数 / 关键词总数。"""
    if not keywords:
        return 0.0
    hit = sum(
        1 for kw in keywords if any(_contains(t, kw) for t in retrieved_texts)
    )
    return hit / len(keywords)

def hit_rate_at_k(keywords: Sequence[str], retrieved_texts: Sequence[str]) -> float:
    """是否至少命中一个关键词（0/1）。"""
    if not keywords:
        return 0.0
    return 1.0 if any(_contains(t, kw) for t in retrieved_texts for kw in keywords) else 0.0

def mrr_at_k(keywords: Sequence[str], retrieved_texts: Sequence[str]) -> float:
    """首个命中关键词所在排名的倒数。"""
    for i, text in enumerate(retrieved_texts, start=1):
        if any(_contains(text, kw) for kw in keywords):
            return 1.0 / i
    return 0.0

def image_hit_rate(gold_cases: list[bool], hits_lists: list[list[dict]]) -> float:
    """图片命中率：image_gold=True 的题里，top-k 命中含图片块的比例。

    hits_lists 每个元素是一道题的命中列表（dict，需带 modality 键），
    与 gold_cases 一一对应。
    """
    total = 0
    ok = 0
    for gold, hits in zip(gold_cases, hits_lists):
        if not gold:
            continue
        total += 1
        if any(h.get("modality") == "image" for h in hits):
            ok += 1
    return ok / total if total else 0.0

def answer_keyword_coverage(reference: str, answer: str) -> float:
    """答案对参考关键词的词面覆盖率。"""
    import re

    terms = set(re.findall(r"[a-z0-9]{2,}", _normalize(reference)))
    terms |= {
        reference[i:i + 2]
        for i in range(len(reference) - 1)
        if all("\u4e00" <= c <= "\u9fff" for c in reference[i:i + 2])
    }
    if not terms:
        return 0.0
    hit = sum(1 for t in terms if t in _normalize(answer))
    return hit / len(terms)

def percentile(values: Sequence[float], pct: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

@dataclass
class CaseResult:
    """单条用例的结果。"""

    question: str
    recall: float = 0.0
    hit: float = 0.0
    mrr: float = 0.0
    coverage: float = 0.0
    latency_ms: float = 0.0
    hit_keywords: list[str] = field(default_factory=list)
    missed_keywords: list[str] = field(default_factory=list)
    answer: str = ""
    route: str = ""

    def to_dict(self) -> dict:
        """单题结果序列化。"""
        return {
            "question": self.question,
            "recall": round(self.recall, 4),
            "hit": self.hit,
            "mrr": round(self.mrr, 4),
            "coverage": round(self.coverage, 4),
            "latency_ms": round(self.latency_ms, 1),
            "hit_keywords": self.hit_keywords,
            "missed_keywords": self.missed_keywords,
            "route": self.route,
        }

@dataclass
class EvalReport:
    """聚合报告。"""

    cases: list[CaseResult] = field(default_factory=list)

    @property
    def size(self) -> int:
        """题目数量。"""
        return len(self.cases)

    def aggregate(self) -> dict[str, float]:
        """汇总指标：各指标取算术平均，另给延迟的均值与 P95。"""
        if not self.cases:
            return {}
        n = len(self.cases)
        lat = [c.latency_ms for c in self.cases]
        return {
            "cases": n,
            "recall": round(sum(c.recall for c in self.cases) / n, 4),
            "hit_rate": round(sum(c.hit for c in self.cases) / n, 4),
            "mrr": round(sum(c.mrr for c in self.cases) / n, 4),
            "coverage": round(sum(c.coverage for c in self.cases) / n, 4),
            "latency_p50_ms": round(percentile(lat, 0.5), 1),
            "latency_p95_ms": round(percentile(lat, 0.95), 1),
        }

    def to_dict(self) -> dict:
        """单题结果序列化。"""
        return {"summary": self.aggregate(),
                "cases": [c.to_dict() for c in self.cases]}

    def to_markdown(self, title: str = "RAG 评测报告") -> str:
        """渲染成 md 表格（写进 reports/ 用）。"""
        agg = self.aggregate()
        if not agg:
            return f"# {title}\n\n（无用例）\n"
        lines = [
            f"# {title}",
            "",
            "| 指标 | 值 |",
            "| --- | --- |",
            f"| 用例数 | {agg['cases']} |",
            f"| 关键词召回率 recall@k | {agg['recall']:.2%} |",
            f"| 命中率 hit_rate@k | {agg['hit_rate']:.2%} |",
            f"| MRR@k | {agg['mrr']:.3f} |",
            f"| 答案关键词覆盖率 | {agg['coverage']:.2%} |",
            f"| 检索延迟 p50 | {agg['latency_p50_ms']:.1f} ms |",
            f"| 检索延迟 p95 | {agg['latency_p95_ms']:.1f} ms |",
            "",
            "## 逐条明细",
            "",
            "| # | 问题 | recall | hit | mrr | coverage | 延迟(ms) | 漏掉关键词 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i, c in enumerate(self.cases, start=1):
            lines.append(
                f"| {i} | {c.question[:40]} | {c.recall:.0%} | {int(c.hit)} | "
                f"{c.mrr:.2f} | {c.coverage:.0%} | {c.latency_ms:.0f} | "
                f"{', '.join(c.missed_keywords) or '-'} |"
            )
        return "\n".join(lines) + "\n"
