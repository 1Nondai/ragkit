"""全局配置（pydantic-settings）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class MilvusSettings(BaseSettings):
    """向量库连接与集合配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    uri: str = Field(
        default="http://127.0.0.1:19530",
        description="Milvus 地址；本地 docker compose 默认 19530。Milvus Lite 可填 './rag.db'",
    )
    collection: str = Field(default="rag_enterprise", description="集合名")
    user: str | None = None
    password: str | None = None
    database: str = "default"

    dense_dim: int = Field(default=1024, description="稠密向量维度，必须与所选 embedding 模型一致")
    rrf_k: int = Field(default=60, ge=1, description="RRF 平滑参数：越小越偏向高排名项")
    strict: bool = Field(
        default=False,
        description="True=连不上 Milvus 直接启动失败；生产必须开，否则会悄悄落到内存库",
    )

    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    search_metric: Literal["IP", "COSINE", "L2"] = "IP"
    consistency_level: Literal["Strong", "Session", "Bounded", "Eventually"] = "Bounded"
    timeout: float = Field(default=20.0, description="单次搜索超时（秒）")

class EmbeddingSettings(BaseSettings):
    """Embedding 提供方。"""

    model_config = SettingsConfigDict(extra="ignore")

    provider: Literal["openai", "hashing", "dashscope-mm"] = Field(
        default="hashing",
        description="openai=OpenAI 兼容文本接口；hashing=离线确定性向量（仅测试/演示）；"
                    "dashscope-mm=阿里云多模态 embedding（图文对齐）",
    )
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    dim_override: int | None = Field(default=None, description="dashscope-mm 词向量用，默认 1024")

class LlmSettings(BaseSettings):
    """大模型配置（OpenAI 兼容协议，DeepSeek / 智谱 / 通义都走这一套）。"""

    model_config = SettingsConfigDict(extra="ignore")

    base_url: str = "https://api.deepseek.com"
    api_key: str | None = None
    model: str = "deepseek-chat"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout: float = Field(default=60.0)

    judge_model: str | None = None

class OcrSettings(BaseSettings):
    """OCR 提供方。"""

    model_config = SettingsConfigDict(extra="ignore")

    provider: Literal["none", "rapidocr", "dots"] = Field(
        default="none",
        description="none=只抽 PDF 文本层；rapidocr=本地 OCR；dots=dots.ocr 版面解析",
    )
    dots_base_url: str = "http://127.0.0.1:6006/v1"
    dots_model: str = "dots_ocr"
    dots_api_key: str | None = None
    dots_max_tokens: int = Field(default=16384, ge=1)
    timeout: float = Field(default=120.0, description="单页 OCR 超时（秒）")

class CaptionSettings(BaseSettings):
    """图片描述配置。描述是增强不是必需：key 没配时自动退化为不生成。"""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=True, description="False=图片块只用 OCR 文字，不发描述请求")
    model: str | None = Field(default=None, description="视觉模型名；None=复用 llm.model")
    base_url: str | None = Field(default=None, description="None=复用 llm.base_url")
    api_key: str | None = Field(default=None, description="None=复用 llm.api_key")
    max_chars: int = Field(default=200, ge=20, le=1000, description="单条描述硬截断长度")
    timeout: float = Field(default=120.0)

class IngestSettings(BaseSettings):
    """数据入库配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    min_chunk_chars: int = Field(default=30, description="过短的块直接丢弃，避免噪声")
    batch_size: int = Field(default=64, ge=1)
    enable_semantic_split: bool = Field(
        default=False,
        description="是否启用语义切分（需要 embedding，成本高；默认关闭走结构+递归切分）",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 800)
        if v >= size:
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则块之间会完全重复")
        return v

class RetrievalSettings(BaseSettings):
    """检索与重排配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    top_k: int = Field(default=5, ge=1, le=50)
    candidate_multiplier: int = Field(
        default=4, ge=1, le=10, description="先多召回、重排后再截断，提升 top_k 质量"
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="低于该分数的候选丢弃")
    enable_rerank: bool = Field(default=False, description="是否启用二次重排（需 rerank 服务）")
    rerank_endpoint: str | None = Field(
        default=None, description="外部重排服务地址；配了就用 ApiReranker，没配用轻量词面重排"
    )
    rerank_model: str = Field(default="bge-reranker-v2-m3", description="外部重排模型名")
    rerank_top_n: int = Field(default=5, ge=1)
    filter_category: str | None = Field(
        default=None, description="按 category 过滤；None 表示不过滤"
    )
    max_query_chars: int = Field(default=500, ge=50)

class GraphSettings(BaseSettings):
    """Corrective / Adaptive RAG 的控制参数。"""

    model_config = SettingsConfigDict(extra="ignore")

    max_rewrite: int = Field(default=2, ge=0, le=5, description="查询改写次数上限")
    max_generate_retry: int = Field(default=1, ge=0, le=3, description="幻觉/答非所问后的重试上限")
    enable_web_fallback: bool = Field(default=False, description="无可用文档时是否走联网检索")
    enable_route: bool = Field(default=True, description="是否启用数据源路由（知识库 vs 联网）")
    min_grounded_score: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="判定'检索够用'的最低覆盖率；太低→频繁重试，太高→直接放弃检索",
    )

class WikiSettings(BaseSettings):
    """Wiki 增强层（可选，默认关闭）。"""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="总开关：关闭时完全不影响原链路")
    root: Path = Field(default=PROJECT_ROOT / "wiki", description="wiki 目录")
    hit_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="路由命中阈值")
    min_coverage: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="内容词覆盖率门槛，低于该值视为未命中",
    )
    stale_days: int = Field(default=90, ge=1, description="volatile 页面多少天算过期")
    writeback_enabled: bool = Field(default=False, description="是否把高价值问答沉淀成草稿")
    writeback_score: float = Field(default=0.7, ge=0.0, le=1.0)

class ApiSettings(BaseSettings):
    """服务层配置。"""

    model_config = SettingsConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8099, ge=1, le=65535)
    api_key: str | None = Field(default=None, description="设置后 /chat 需带 X-API-Key")
    cors_origins: str = "*"

class Settings(BaseSettings):
    """聚合配置。env 前缀 RAG_，支持 .env 文件。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="RAG_",
        env_nested_delimiter="__", extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    milvus: MilvusSettings = MilvusSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LlmSettings = LlmSettings()
    ingest: IngestSettings = IngestSettings()
    ocr: OcrSettings = OcrSettings()
    caption: CaptionSettings = CaptionSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    wiki: WikiSettings = WikiSettings()
    graph: GraphSettings = GraphSettings()
    api: ApiSettings = ApiSettings()

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例。用 lru_cache 而不是模块级实例，测试里可以 clear_cache 后重载。"""
    return Settings()

settings = get_settings()

def reload_settings() -> Settings:
    """重新读取配置（测试与热加载用）。"""
    get_settings.cache_clear()
    return get_settings()

def redacted() -> dict:
    """可安全打印的配置快照（密钥打码）。用于 /health 与日志。"""
    s = get_settings()
    return {
        "env": s.env,
        "log_level": s.log_level,
        "milvus": {"uri": s.milvus.uri, "collection": s.milvus.collection,
                   "dense_dim": s.milvus.dense_dim, "strict": s.milvus.strict},
        "embedding": {"provider": s.embedding.provider, "model": s.embedding.model},
        "llm": {"model": s.llm.model, "base_url": s.llm.base_url,
                "api_key_set": bool(s.llm.api_key)},
        "ingest": {"chunk_size": s.ingest.chunk_size, "overlap": s.ingest.chunk_overlap},
        "ocr": {"provider": s.ocr.provider, "dots_model": s.ocr.dots_model},
        "caption": {"enabled": s.caption.enabled, "model": s.caption.model or s.llm.model},
        "retrieval": {"top_k": s.retrieval.top_k, "rerank": s.retrieval.enable_rerank},
        "graph": {"max_rewrite": s.graph.max_rewrite, "web_fallback": s.graph.enable_web_fallback},
        "wiki": {"enabled": s.wiki.enabled, "root": str(s.wiki.root),
                 "hit_threshold": s.wiki.hit_threshold,
                 "min_coverage": s.wiki.min_coverage},
    }

def _env_override_demo() -> None:
    """打印当前生效的环境变量（排查'我明明配了却不生效'）。"""
    for k, v in sorted(os.environ.items()):
        if k.startswith("RAG_"):
            print(f"{k}={v}")
