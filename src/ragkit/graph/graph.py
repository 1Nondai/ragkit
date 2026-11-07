"""图编排：把节点串成 Corrective / Adaptive RAG 流程。"""

from __future__ import annotations

from functools import partial
from typing import Any

from ragkit.graph.nodes import (
    Deps,
    node_chitchat,
    node_clarify,
    node_generate,
    node_grade,
    node_retrieve,
    node_rewrite,
    node_route,
    node_verify,
    node_wiki,
)
from ragkit.graph.state import RAGState, new_state
from ragkit.logging_setup import get_logger

log = get_logger("graph.builder")

END = "__end__"

def route_next(state: dict[str, Any]) -> str:
    """路由后的分支：闲聊/追问直连，其余先试 wiki 再走检索。"""
    route = state.get("route") or "kb"
    if route == "chitchat":
        return "chitchat"
    if route == "clarify":
        return "clarify"
    return "wiki"

def wiki_next(state: dict[str, Any]) -> str:
    """wiki 命中就结束（已由 wiki 节点填好答案），未命中回检索链路。"""
    return END if state.get("wiki_hit") else "rewrite"

def grade_next(state: dict[str, Any], max_rewrite: int = 2) -> str:
    """检索质量评估后的分支：够用就生成，不够用且没到上限就改写重试。"""
    if state.get("grounded"):
        return "generate"
    if int(state.get("rewrite_count", 0)) < max_rewrite:
        log.info("检索结果不足，触发第 %d 次改写重试", state.get("rewrite_count", 0) + 1)
        return "rewrite"
    return "generate"

def verify_next(state: dict[str, Any], max_retry: int = 1) -> str:
    """幻觉校验后的分支：不通过且还有重试额度就重生成。"""
    if state.get("grounded"):
        return END
    retries = int(state.get("generate_retry", 0))
    if retries < max_retry:
        return "generate"
    return END

def build_graph(deps: Deps):
    """编译 LangGraph 图；langgraph 不可用时回落到顺序执行器（保证可运行）。"""
    try:
        from langgraph.graph import END as LG_END
        from langgraph.graph import StateGraph
    except Exception as exc:  # noqa: BLE001
        log.warning("langgraph 不可用，回落到顺序执行器: %s", exc)
        return SimpleRunner(deps)

    max_rewrite = deps.settings.graph.max_rewrite
    max_retry = deps.settings.graph.max_generate_retry

    graph = StateGraph(RAGState)
    graph.add_node("route", partial(node_route, deps=deps))
    graph.add_node("rewrite", partial(node_rewrite, deps=deps))
    graph.add_node("retrieve", partial(node_retrieve, deps=deps))
    graph.add_node("grade", partial(node_grade, deps=deps))
    graph.add_node("generate", partial(node_generate, deps=deps))
    graph.add_node("verify", partial(node_verify, deps=deps))
    graph.add_node("wiki", partial(node_wiki, deps=deps))
    graph.add_node("chitchat", partial(node_chitchat, deps=deps))
    graph.add_node("clarify", partial(node_clarify, deps=deps))

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route", route_next,
        {"chitchat": "chitchat", "clarify": "clarify", "wiki": "wiki"},
    )
    graph.add_conditional_edges("wiki", wiki_next, {END: END, "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", partial(grade_next, max_rewrite=max_rewrite),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify", partial(verify_next, max_retry=max_retry),
        {"generate": "generate", LG_END: LG_END},
    )
    graph.add_edge("chitchat", LG_END)
    graph.add_edge("clarify", LG_END)

    return graph.compile()

class SimpleRunner:
    """顺序执行器：复现同样的控制流，不依赖 langgraph。

    存在的意义：langgraph 是重依赖（装一次要拉一堆包），
    CI 或精简环境里可能没有。有它就能保证"至少能跑通"。
    """

    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """复现同一套控制流：路由 → wiki → 改写/检索循环 → 生成/校验循环。"""
        state = dict(state)
        max_rewrite = self.deps.settings.graph.max_rewrite
        max_retry = self.deps.settings.graph.max_generate_retry

        state.update(node_route(state, self.deps))
        branch = route_next(state)
        if branch == "chitchat":
            state.update(node_chitchat(state, self.deps))
            return state
        if branch == "clarify":
            state.update(node_clarify(state, self.deps))
            return state

        state.update(node_wiki(state, self.deps))
        if state.get("wiki_hit"):
            return state

        for _ in range(max_rewrite + 1):
            state.update(node_rewrite(state, self.deps))
            state.update(node_retrieve(state, self.deps))
            state.update(node_grade(state, self.deps))
            if grade_next(state, max_rewrite) == "generate":
                break

        retries = 0
        while True:
            state.update(node_generate(state, self.deps))
            state.update(node_verify(state, self.deps))
            if state.get("grounded") or retries >= max_retry:
                break
            retries += 1
            state["generate_retry"] = retries
        return state

def run(question: str, deps: Deps, *, session_id: str = "default",
        kb_name: str | None = None, compiled=None) -> dict[str, Any]:
    """一次问答的完整入口。

    用法：state = run("Milvus 是什么", deps)
          state["answer"] / state["citations"] / state["timings"]
    """
    app = compiled or build_graph(deps)
    state = dict(new_state(question, session_id=session_id, kb_name=kb_name))
    return app.invoke(state)
