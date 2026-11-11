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


def _trace_wrapped(name: str, fn, deps: Deps):
    """把节点包一层观测 span；未启用观测时零开销直通，节点函数对观测零感知。"""
    sink = getattr(deps, "trace_sink", None)

    def wrapper(state: dict[str, Any]):
        if sink is None or not getattr(sink, "active", False):
            return fn(state, deps)
        with sink.node_span(name):
            return fn(state, deps)

    return wrapper

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
    graph.add_node("route", _trace_wrapped("route", node_route, deps))
    graph.add_node("rewrite", _trace_wrapped("rewrite", node_rewrite, deps))
    graph.add_node("retrieve", _trace_wrapped("retrieve", node_retrieve, deps))
    graph.add_node("grade", _trace_wrapped("grade", node_grade, deps))
    graph.add_node("generate", _trace_wrapped("generate", node_generate, deps))
    graph.add_node("verify", _trace_wrapped("verify", node_verify, deps))
    graph.add_node("wiki", _trace_wrapped("wiki", node_wiki, deps))
    graph.add_node("chitchat", _trace_wrapped("chitchat", node_chitchat, deps))
    graph.add_node("clarify", _trace_wrapped("clarify", node_clarify, deps))

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

    def _step(self, name: str, fn, state: dict[str, Any]) -> dict[str, Any]:
        """顺序执行一个节点；与 build_graph 一样在观测启用时包一层 span。"""
        sink = getattr(self.deps, "trace_sink", None)
        if sink is None or not getattr(sink, "active", False):
            return fn(state, self.deps)
        with sink.node_span(name):
            return fn(state, self.deps)

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """复现同一套控制流：路由 → wiki → 改写/检索循环 → 生成/校验循环。"""
        state = dict(state)
        max_rewrite = self.deps.settings.graph.max_rewrite
        max_retry = self.deps.settings.graph.max_generate_retry

        state.update(self._step("route", node_route, state))
        branch = route_next(state)
        if branch == "chitchat":
            state.update(self._step("chitchat", node_chitchat, state))
            return state
        if branch == "clarify":
            state.update(self._step("clarify", node_clarify, state))
            return state

        state.update(self._step("wiki", node_wiki, state))
        if state.get("wiki_hit"):
            return state

        for _ in range(max_rewrite + 1):
            state.update(self._step("rewrite", node_rewrite, state))
            state.update(self._step("retrieve", node_retrieve, state))
            state.update(self._step("grade", node_grade, state))
            if grade_next(state, max_rewrite) == "generate":
                break

        retries = 0
        while True:
            state.update(self._step("generate", node_generate, state))
            state.update(self._step("verify", node_verify, state))
            if state.get("grounded") or retries >= max_retry:
                break
            retries += 1
            state["generate_retry"] = retries
        return state

def run(question: str, deps: Deps, *, session_id: str = "default",
        kb_name: str | None = None, groups: list[str] | None = None,
        compiled=None) -> dict[str, Any]:
    """一次问答的完整入口。"""
    app = compiled or build_graph(deps)
    state = dict(new_state(question, session_id=session_id, kb_name=kb_name, groups=groups))

    sink = getattr(deps, "trace_sink", None)
    if sink is None or not getattr(sink, "active", False):
        return app.invoke(state)

    with sink.request_span(state):
        try:
            result = app.invoke(state)
        except Exception as exc:  # noqa: BLE001
            sink.record_error(exc)
            raise
        sink.record_result(result)
        return result
