"""图的状态定义。"""

from __future__ import annotations

from typing import Any, TypedDict

MAX_REWRITE_ATTEMPTS = 3

class RAGState(TypedDict, total=False):
    """RAG 图的状态。所有字段都是可选的（节点按需读写）。"""

    question: str
    session_id: str
    user_id: str
    kb_name: str | None
    route_kb: str | None

    rewritten_queries: list[str]
    effective_query: str
    rewrite_count: int
    route: str

    wiki_hit: bool
    wiki_slug: str | None
    wiki_score: float
    wiki_gate: str
    wiki_reason: str

    hits: list[dict[str, Any]]
    context: str
    citations: list[dict[str, Any]]
    retrieval_elapsed_ms: float
    degraded: bool

    answer: str
    grounded: bool
    confidence: float
    generate_retry: int

    timings: dict[str, float]
    trace: list[str]
    errors: list[str]
    warnings: list[str]

def new_state(question: str, *, session_id: str = "default", user_id: str = "anonymous",
              kb_name: str | None = None) -> RAGState:
    """构造初始状态（不要直接写 dict 字面量，容易漏字段）。"""
    return RAGState(
        question=question,
        session_id=session_id,
        user_id=user_id,
        kb_name=kb_name,
        route_kb=None,
        wiki_hit=False,
        wiki_slug=None,
        wiki_score=0.0,
        wiki_gate="",
        wiki_reason="",
        rewritten_queries=[],
        effective_query=question,
        rewrite_count=0,
        route="",
        hits=[],
        context="",
        citations=[],
        retrieval_elapsed_ms=0.0,
        degraded=False,
        answer="",
        grounded=False,
        confidence=0.0,
        generate_retry=0,
        timings={},
        trace=[],
        errors=[],
        warnings=[],
    )
