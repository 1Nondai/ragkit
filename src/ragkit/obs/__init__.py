"""obs：可选观测层（Langfuse）。默认关闭；SDK 未安装时自动降级为 NoopSink。"""

from ragkit.obs.langfuse_sink import LangfuseSink, ObservableChat, build_trace_sink
from ragkit.obs.protocol import NoopSink, TraceSink

__all__ = [
    "LangfuseSink",
    "NoopSink",
    "ObservableChat",
    "TraceSink",
    "build_trace_sink",
]
