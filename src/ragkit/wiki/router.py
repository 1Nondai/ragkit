"""Wiki 路由：先问已编译的知识，命中就直答，不命中再走检索。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from ragkit.wiki.schema import WikiPage


@dataclass
class WikiDecision:

    hit: bool
    slug: str | None = None
    title: str | None = None
    score: float = 0.0
    reason: str = ""
    gate: str = ""
    coverage: float = 0.0
    page: WikiPage | None = None

    def to_dict(self) -> dict[str, object]:
        """转成可打印/可日志的判断结果。"""
        return {
            "hit": self.hit, "slug": self.slug, "title": self.title,
            "score": round(self.score, 4), "coverage": round(self.coverage, 4),
            "reason": self.reason, "gate": self.gate,
        }

def _bigrams(text: str) -> set:
    text = (text or "").lower().strip()
    return {text[i:i + 2] for i in range(len(text) - 1)}

def _jaccard(a: str, b: str) -> float:
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)

_STOP_CHARS = set("的是了么呢吗我你他它这那有什么如和对与及在为就都也很要会能可以到把被从向")
_EN_STOP = {"the", "and", "for", "with", "what", "how", "why", "does", "can", "are", "was"}

def content_terms(text: str) -> set[str]:
    """查询的"内容词"：英文/数字词（去停用词）+ 非虚词构成的中文双字。"""
    q = (text or "").lower()
    terms = {t for t in re.findall(r"[a-z0-9]{3,}", q) if t not in _EN_STOP}
    terms |= {
        q[i:i + 2]
        for i in range(len(q) - 1)
        if all("\u4e00" <= c <= "\u9fff" for c in q[i:i + 2])
        and not (q[i] in _STOP_CHARS or q[i + 1] in _STOP_CHARS)
    }
    return terms

def content_coverage(page: WikiPage, query: str) -> float:
    """查询内容词在页面里的覆盖率（标题 + 别名 + 正文一起算）。"""
    terms = content_terms(query)
    if not terms:
        return 0.0
    haystack = " ".join([page.title, *page.aliases, page.body]).lower()
    return round(sum(1 for t in terms if t in haystack) / len(terms), 4)

def score_page(page: WikiPage, query: str) -> float:
    """页面与查询的相关度（0~1）。顺序很重要：包含优先，bigram 兜底。"""
    q = (query or "").lower().strip()
    title = (page.title or "").lower().strip()
    if not q or not title:
        return 0.0

    if title in q or q in title:
        return 1.0

    best = _jaccard(q, title)

    for alias in page.aliases:
        a = alias.lower().strip()
        if not a:
            continue
        if a in q or q in a:
            best = max(best, 0.95)
        else:
            best = max(best, _jaccard(q, a) * 0.95)

    tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", q)}
    if tokens:
        body = page.body.lower()
        coverage = sum(1 for t in tokens if t in body) / len(tokens)
        best = max(best, min(0.7, coverage * 0.9))
    return round(best, 4)

class WikiRouter:
    """把"该不该用 wiki 直答"变成一次可解释的判断。"""

    def __init__(
        self,
        pages: Sequence[WikiPage],
        *,
        threshold: float = 0.6,
        stale_days: int = 90,
        today: str | None = None,
        min_coverage: float = 0.2,
    ) -> None:
        self.pages = list(pages)
        self.threshold = threshold
        self.stale_days = stale_days
        self.today = today or date.today().isoformat()
        self.min_coverage = min_coverage

    def candidates(self, query: str) -> list[tuple[float, WikiPage]]:
        """按相关度排序的候选页（score=0 的不进候选）。"""
        scored = [(score_page(p, query), p) for p in self.pages]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def decide(self, query: str) -> WikiDecision:
        """四道门依次判定，任一不过就降级到 RAG。

        用法：d = router.decide("Milvus 是什么")
              d.hit / d.gate（被哪道门拦下）/ d.reason（人话原因）
        """
        if not (query or "").strip():
            return WikiDecision(False, reason="空查询")
        if not self.pages:
            return WikiDecision(False, reason="wiki 为空（先跑 scripts/build_wiki.py）")

        scored = self.candidates(query)
        if not scored:
            return WikiDecision(False, reason="没有可比对的页面")

        score, page = scored[0]
        coverage = content_coverage(page, query)
        gate = self._gate(page, score, coverage)
        if gate:
            return WikiDecision(False, slug=page.slug, title=page.title, score=score,
                                gate=gate, reason=self._reason(gate, score, coverage),
                                coverage=coverage)

        return WikiDecision(True, slug=page.slug, title=page.title, score=score,
                            reason="命中已编译知识页", page=page, coverage=coverage)

    def _gate(self, page: WikiPage, score: float, coverage: float = 1.0) -> str:
        """按顺序返回被拦下的门名；空串表示全部通过。"""
        if score < self.threshold:
            return "score"
        if coverage < self.min_coverage:
            return "content"
        if page.confidence == "low":
            return "confidence"
        if page.contested:
            return "contested"
        if page.is_stale(self.today, self.stale_days):
            return "stale"
        return ""

    def _reason(self, gate: str, score: float, coverage: float) -> str:
        return {
            "score": f"相似度 {score:.2f} 低于阈值 {self.threshold}",
            "content": f"内容词覆盖率 {coverage:.2f} 低于阈值 {self.min_coverage}（页面答不了这个问题）",
            "confidence": "页面置信度 low（未校订），退回检索",
            "contested": "页面存在未解冲突，退回检索",
            "stale": f"易腐化页面超过 {self.stale_days} 天未更新，退回检索",
        }.get(gate, "未命中")

    @staticmethod
    def compose_answer(page: WikiPage) -> str:
        """直答**只做组装**，不再调模型。"""
        parts = [page.body.strip()]
        if page.sources:
            parts.append("\n\n---\n\n**出处**：" + "；".join(page.sources[:5]))
        return "\n".join(parts)

@dataclass
class WikiHitStats:
    """批量评估命中率用的小容器（不落盘，只算）。"""

    total: int = 0
    hits: int = 0
    gates: dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """批量评估时的命中率。"""
        return round(self.hits / self.total, 4) if self.total else 0.0
