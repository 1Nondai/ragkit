"""接口契约（pydantic 模型）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求体：question 必填，session_id 用于多轮，kb_name 为空则自动路由。"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID（多轮用）")
    kb_name: str | None = Field(default=None, description="指定知识库；None=自动路由")
    groups: list[str] | None = Field(
        default=None, description="调用方权限组；传入后只检索 acl 有交集的块（不传=不过滤）"
    )

class Citation(BaseModel):
    """引用项：答案依据的文档片段（带 image_path 时前端可回显原图）。"""
    source: str = ""
    title: str = ""
    chunk_id: str = ""
    score: float = 0.0
    image_path: str = ""

class Hit(BaseModel):
    """检索命中摘要：给 /chat 响应里的 hits 字段用。"""
    source: str = ""
    title: str = ""
    score: float = 0.0
    image_path: str = ""

class ChatResponse(BaseModel):
    """问答响应体：答案 + 引用 + 命中 + 路由/置信度 + 分阶段耗时。"""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    route: str = ""
    grounded: bool = False
    confidence: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: list[str] = Field(default_factory=list)

class SearchRequest(BaseModel):
    """检索请求体：只检索不生成（调试与前端自拼答案用）。"""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    groups: list[str] | None = Field(default=None, description="权限组（acl 前置过滤）")

class SearchImageRequest(BaseModel):
    """图搜图：image_base64 为纯 base64（不带 data: 前缀）。"""

    image_base64: str = Field(..., min_length=1, description="图片的 base64 编码")
    top_k: int | None = Field(default=None, ge=1, le=50)

class SearchResponse(BaseModel):
    """检索响应体：命中列表 + 检索耗时。"""
    query: str
    hits: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0

class IngestRequest(BaseModel):
    """入库请求体：指定目录与是否重建集合。"""
    data_dir: str = Field(default="data/raw", description="要入库的文档目录")
    recreate: bool = Field(default=False, description="是否删集合重建（换模型时必须 True）")

class IngestResponse(BaseModel):
    """入库统计（字段与 IngestStats.to_dict 对齐）。"""
    files: int = 0
    sections: int = 0
    chunks: int = 0
    written: int = 0
    skipped_files: int = 0
    elapsed_ms: float = 0.0
    collection: str = ""
    embedded_dim: int = 0

class HealthResponse(BaseModel):
    """健康检查响应：库类型、文档数、脱敏配置快照。"""
    status: str = "ok"
    env: str = "development"
    store: str = ""
    doc_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
