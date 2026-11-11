"""观测协议层：TraceSink 接口与永不阻断主链路的 NoopSink（观测是旁路，失败一律降级）。"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from ragkit.logging_setup import get_logger

log = get_logger("obs.protocol")

class TraceSink(Protocol):
    """观测汇接口：request/node/generation 三层 span + 结果打分。

    方法都是上下文管理器或一次性动作，调用方用 `with` 进入/退出，
    span 的父子关系由实现方负责（Langfuse 侧靠 OTel 上下文自动挂树）。
    """

    active: bool

    def request_span(self, state: dict[str, Any]) -> AbstractContextManager[Any]:
        """请求级根 span。with 块内执行整张图；退出时实现方决定是否补记录。"""
        ...

    def node_span(self, name: str) -> AbstractContextManager[Any]:
        """节点级 span（route / retrieve / ...）。"""
        ...

    def generation_span(
        self, *, name: str, input: Any, model: str | None, metadata: dict | None = None
    ) -> AbstractContextManager[Any]:
        """模型调用级 generation，yield 出的对象可 update(output=...)。"""
        ...

    def record_result(self, state: dict[str, Any]) -> None:
        """请求成功结束：把答案/元数据/分数写到当前 trace 上。"""
        ...

    def record_error(self, exc: BaseException) -> None:
        """请求失败：给当前 span 打 ERROR 级别，原样抛出由实现方之外处理。"""
        ...

    def flush(self) -> None:
        """把缓冲的上报数据推给服务端（进程退出前也应调用）。"""
        ...

class NoopSink:
    """空实现：什么都不记，只为让调用方不用写 if 判断。

    三个场景会用到它：配置未启用、缺少密钥、langfuse SDK 不可用/初始化失败。
    """

    active = False

    @contextmanager
    def request_span(self, state: dict[str, Any]) -> AbstractContextManager[None]:
        yield None

    @contextmanager
    def node_span(self, name: str) -> AbstractContextManager[None]:
        yield None

    @contextmanager
    def generation_span(
        self, *, name: str, input: Any, model: str | None, metadata: dict | None = None
    ) -> AbstractContextManager[None]:
        yield None

    def record_result(self, state: dict[str, Any]) -> None:
        return None

    def record_error(self, exc: BaseException) -> None:
        return None

    def flush(self) -> None:
        return None
