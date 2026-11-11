"""Langfuse 观测汇（langfuse>=5 SDK）：request/node/generation 三层 span + 结果打分，任何上报故障都不影响回答。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ragkit.config import ObsSettings, get_settings
from ragkit.logging_setup import get_logger
from ragkit.obs.protocol import NoopSink, TraceSink

log = get_logger("obs.langfuse")

class LangfuseSink:
    """TraceSink 的 Langfuse 实现（一个进程一个实例，随 Deps 复用）。"""

    def __init__(self, obs: ObsSettings) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=obs.public_key,
            secret_key=obs.secret_key,
            host=obs.host,
            environment=obs.environment,
            sample_rate=obs.sample_rate,
            tracing_enabled=True,
        )
        self.active = True
        self._flush_on_finish = obs.flush_on_finish

    @staticmethod
    def _set_trace_attrs(obs: Any, *, session_id: str, user_id: str, tags: list[str]) -> None:
        try:
            otel = getattr(obs, "_otel_span", None)
            if otel is None:
                return
            otel.set_attribute("session.id", session_id)
            otel.set_attribute("user.id", user_id)
            otel.set_attribute("langfuse.trace.tags", tags)
        except Exception as exc:  # noqa: BLE001 - 观测属性失败不阻断主链路
            log.debug("trace 属性写入失败（不影响主链路）: %s", exc)

    @contextmanager
    def request_span(self, state: dict[str, Any]) -> Iterator[Any]:
        """请求级根 span：名字固定 rag-query，输入是原始问题。

        退出时 OTel 自动结束 span；业务异常会原样传播（不吞），
        错误标记由 run() 里的 record_error 负责。
        """
        session_id = str(state.get("session_id") or "default")
        user_id = str(state.get("user_id") or "anonymous")
        kb = state.get("kb_name")
        root = self._client.start_as_current_observation(
            name="rag-query",
            as_type="span",
            input={"question": state.get("question", "")},
            metadata={"kb_name": kb, "groups": state.get("user_groups") or []},
        )
        self._set_trace_attrs(
            root,
            session_id=session_id,
            user_id=user_id,
            tags=["ragkit", f"kb:{kb}" if kb else "kb:auto"],
        )
        with root:
            yield root

    @contextmanager
    def node_span(self, name: str) -> Iterator[None]:
        """节点级 span（as_type=chain）。父级=当时所处的 OTel 上下文。"""
        with self._client.start_as_current_observation(name=name, as_type="chain"):
            yield

    @contextmanager
    def generation_span(
        self, *, name: str, input: Any, model: str | None, metadata: dict | None = None
    ) -> Iterator[Any]:
        """模型调用级 generation：自动记输入、模型名与耗时。"""
        with self._client.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input, metadata=metadata
        ) as gen:
            yield gen

    @staticmethod
    def _summary(state: dict[str, Any]) -> dict[str, Any]:
        """trace 的 metadata 摘要：只放小字段，context/长文本不进 metadata。"""
        return {
            "route": state.get("route", ""),
            "grounded": state.get("grounded", False),
            "confidence": state.get("confidence", 0.0),
            "wiki_hit": state.get("wiki_hit", False),
            "degraded": state.get("degraded", False),
            "rewrite_count": state.get("rewrite_count", 0),
            "generate_retry": state.get("generate_retry", 0),
            "hits": len(state.get("hits", [])),
            "timings_ms": state.get("timings", {}),
            "trace": state.get("trace", []),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }

    def _score(self, name: str, value: float, data_type: str) -> None:
        try:
            self._client.score_current_trace(name=name, value=value, data_type=data_type)
        except Exception as exc:  # noqa: BLE001
            log.debug("score %s 写入失败（不影响主链路）: %s", name, exc)

    def record_result(self, state: dict[str, Any]) -> None:
        """成功收尾：output=答案+引用，metadata=全链路摘要，并打 4 类分数。"""
        try:
            self._client.update_current_span(
                output={
                    "answer": state.get("answer", ""),
                    "citations": state.get("citations", []),
                },
                metadata=self._summary(state),
            )
            self._score("confidence", round(float(state.get("confidence") or 0.0), 4), "NUMERIC")
            self._score("grounded", 1.0 if state.get("grounded") else 0.0, "BOOLEAN")
            self._score("wiki_hit", 1.0 if state.get("wiki_hit") else 0.0, "BOOLEAN")
            if state.get("degraded"):
                self._score("degraded", 1.0, "BOOLEAN")
            timings = state.get("timings") or {}
            total_ms = sum(float(v or 0) for v in timings.values())
            if total_ms:
                self._score("latency_ms", round(total_ms, 1), "NUMERIC")
            if self._flush_on_finish:
                self._client.flush()
        except Exception as exc:  # noqa: BLE001 - 上报失败绝不影响返回结果
            log.debug("record_result 失败（不影响主链路）: %s", exc)

    def record_error(self, exc: BaseException) -> None:
        """失败收尾：当前 span 标 ERROR，score 仍打（grounded=0 等）。"""
        try:
            self._client.update_current_span(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            )
            self._score("grounded", 0.0, "BOOLEAN")
            if self._flush_on_finish:
                self._client.flush()
        except Exception as inner:  # noqa: BLE001
            log.debug("record_error 失败: %s", inner)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            log.debug("flush 失败: %s", exc)

class ObservableChat:
    """ChatModel 包装器：把每次模型调用记成一条 generation。

    只在有真实模型时启用包装（ExtractiveLLM 离线实现不包——
    Deps.offline 用 isinstance 判断离线，包一层会破坏这个判断）。
    其余属性通过 __getattr__ 透传（supports_vision 等照常工作）。
    """

    def __init__(self, inner: Any, sink: TraceSink) -> None:
        self._inner = inner
        self._sink = sink
        self.model = getattr(inner, "model", None)

    def invoke(self, prompt: str) -> str:
        if not self._sink.active:
            return self._inner.invoke(prompt)
        return self._observe("llm.invoke", input=prompt, call=self._inner.invoke, arg=prompt)

    def invoke_messages(self, messages: list[dict]) -> str:
        if not self._sink.active:
            return self._inner.invoke_messages(messages)
        return self._observe(
            "llm.invoke_messages", input=messages, call=self._inner.invoke_messages, arg=messages
        )

    def _observe(self, name: str, *, input: Any, call: Any, arg: Any) -> str:
        started = time.perf_counter()
        try:
            with self._sink.generation_span(name=name, input=input, model=self.model) as gen:
                output = call(arg)
                gen.update(
                    output=output,
                    metadata={"latency_ms": round((time.perf_counter() - started) * 1000, 1)},
                )
                return output
        except Exception as exc:  # noqa: BLE001 - 记录失败不影响调用本身
            log.debug("generation 记录中断（模型调用本身可能仍成功）: %s", exc)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

def build_trace_sink(settings: Any = None) -> TraceSink:
    """按配置构造观测汇；任何一步不满足都降级为 NoopSink。

    降级路径（按顺序检查）：
      1. obs.enabled=False → NoopSink（正常默认，不打扰任何人）；
      2. 缺 public_key / secret_key → NoopSink + 警告；
      3. langfuse SDK 未安装或初始化失败 → NoopSink + 警告。
    CI 里故意不安装 langfuse，就是为了持续验证第 3 条降级真实可用。
    """
    settings = settings or get_settings()
    obs = settings.obs
    if not obs.enabled:
        log.info("Langfuse 未启用（obs.enabled=False），不采集观测数据")
        return NoopSink()
    if not obs.public_key or not obs.secret_key:
        log.warning("Langfuse 已启用但缺少 public_key/secret_key，观测降级为无操作")
        return NoopSink()
    try:
        sink: TraceSink = LangfuseSink(obs)
        log.info("Langfuse 观测已启用（host=%s, environment=%s）", obs.host, obs.environment)
        return sink
    except Exception as exc:  # noqa: BLE001 - SDK 缺失/配置错误都要降级
        log.warning("Langfuse 初始化失败，观测降级为无操作: %s", exc)
        return NoopSink()
