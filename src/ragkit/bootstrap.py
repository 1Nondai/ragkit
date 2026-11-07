"""组装入口：把配置 → 各层实现 → 图，一次性拼好。"""

from __future__ import annotations

from ragkit.config import Settings, get_settings
from ragkit.embeddings import Embeddings, build_embeddings
from ragkit.graph.graph import run
from ragkit.graph.nodes import Deps, HistoryStore
from ragkit.graph.router import HybridRouter, KeywordRouter, LlmRouter
from ragkit.ingest.pipeline import IngestPipeline
from ragkit.llm import build_chat_model
from ragkit.logging_setup import get_logger
from ragkit.retrieval.protocol import VectorStore
from ragkit.retrieval.service import RetrievalService

log = get_logger("bootstrap")

_HISTORY = HistoryStore()
_MEMORY_STORE: VectorStore | None = None
_WIKI_CACHE: dict = {}

def build_store(settings: Settings | None = None, *, prefer_memory: bool = False) -> VectorStore:
    """按配置构造向量库；连不上 Milvus 时自动回落到内存库（并告警）。"""
    settings = settings or get_settings()
    uri = (settings.milvus.uri or "").strip()
    global _MEMORY_STORE
    if prefer_memory or uri.startswith("memory"):
        if _MEMORY_STORE is None:
            from ragkit.retrieval.protocol import InMemoryStore

            log.warning("使用内存向量库，数据不会持久化（uri=%s）", uri or "prefer_memory")
            _MEMORY_STORE = InMemoryStore()
        return _MEMORY_STORE

    from ragkit.retrieval.milvus_store import MilvusStore

    m = settings.milvus
    store = MilvusStore(
        uri=m.uri, collection=m.collection, user=m.user, password=m.password,
        database=m.database, timeout=m.timeout, metric=m.search_metric,
        consistency_level=m.consistency_level,
    )
    try:
        store.client()
        log.info("已连接 Milvus: %s / 集合 %s", m.uri, m.collection)
        return store
    except Exception as exc:  # noqa: BLE001
        if m.strict:
            raise RuntimeError(f"连接 Milvus 失败（strict 模式）: {m.uri} -> {exc}") from exc
        log.warning("连接 Milvus 失败（%s），回落到内存向量库: %s", m.uri, exc)
        from ragkit.retrieval.protocol import InMemoryStore

        return InMemoryStore()

def build_service(store: VectorStore, embeddings: Embeddings,
                  settings: Settings | None = None) -> RetrievalService:
    """用配置里的 top_k / 候选倍数 / 阈值 / rrf_k 组装检索服务。"""
    settings = settings or get_settings()
    r = settings.retrieval
    return RetrievalService(
        store=store, embeddings=embeddings, top_k=r.top_k,
        candidate_multiplier=r.candidate_multiplier, score_threshold=r.score_threshold,
        rrf_k=settings.milvus.rrf_k, filter_category=r.filter_category,
    )

def build_router(chat, settings: Settings | None = None) -> HybridRouter:
    """规则优先 + LLM 兜底的两级路由器（enable_llm 可关）。"""
    settings = settings or get_settings()
    return HybridRouter(
        keyword_router=KeywordRouter(),
        llm_router=LlmRouter(chat),
        enable_llm=settings.graph.enable_route,
    )

def build_wiki(settings: Settings | None = None):
    """构造 wiki 管理器与路由器（进程内缓存；页面重建后要 reset_wiki）。"""
    settings = settings or get_settings()
    key = str(settings.wiki.root)
    cached = _WIKI_CACHE.get(key)
    if cached is not None:
        return cached

    from ragkit.wiki.manager import WikiManager
    from ragkit.wiki.router import WikiRouter

    manager = WikiManager(settings.wiki.root)
    pages = manager.load() if manager.concepts_dir.exists() else []
    router = WikiRouter(pages, threshold=settings.wiki.hit_threshold,
                        stale_days=settings.wiki.stale_days,
                        min_coverage=settings.wiki.min_coverage)
    _WIKI_CACHE[key] = (manager, router)
    log.info("wiki 层已加载: %d 页（%s）", len(pages), manager.root)
    return manager, router

def reset_wiki() -> None:
    """清缓存（重建 wiki 之后必须调用，否则仍用旧页面）。"""
    _WIKI_CACHE.clear()

def build_deps(settings: Settings | None = None, *, prefer_memory: bool = False) -> Deps:
    """装配完整依赖（脚本、API、测试统一走这里）。

    用法：deps = build_deps(prefer_memory=True)  # 进程内复用同一套实例
    """
    settings = settings or get_settings()
    embeddings = build_embeddings()
    store = build_store(settings, prefer_memory=prefer_memory)
    service = build_service(store, embeddings, settings)
    chat = build_chat_model()
    deps = Deps(
        service=service, chat=chat, router=build_router(chat, settings),
        settings=settings, history=_HISTORY,
        image_root=settings.ingest.data_dir.parent,
    )
    if settings.wiki.enabled:
        deps.wiki_manager, deps.wiki_router = build_wiki(settings)
    return deps

def build_ocr_client(settings: Settings | None = None):
    """按配置构造 OCR 客户端；provider=none 时返回 None（PDF 走文本层）。"""
    settings = settings or get_settings()
    from ragkit.multimodal.ocr_client import build_ocr

    s = settings.ocr
    kwargs = {}
    if s.provider == "dots":
        kwargs = dict(
            base_url=s.dots_base_url, model=s.dots_model,
            api_key=s.dots_api_key, timeout=s.timeout, max_tokens=s.dots_max_tokens,
        )
    return build_ocr(s.provider, **kwargs)

def build_captioner(settings: Settings | None = None):
    """按配置构造图片描述器；没 key / 显式关闭时退化为不生成（NullCaptioner）。"""
    settings = settings or get_settings()
    from ragkit.multimodal.captioner import CachedCaptioner, LlmCaptioner, NullCaptioner

    s = settings.caption
    if not s.enabled:
        log.info("图片描述已关闭（caption.enabled=False）")
        return NullCaptioner()
    api_key = s.api_key or settings.llm.api_key
    if not api_key:
        log.info("未配置视觉模型 key，图片描述退化为不生成")
        return NullCaptioner()
    model = s.model or settings.llm.model
    if s.model is None and settings.llm.model == "deepseek-chat":
        log.warning("caption.model 未单独配置，复用的是纯文本模型 %s，图片描述会失败；"
                    "建议配置视觉模型名（RAG_CAPTION__MODEL）", model)
    inner = LlmCaptioner(
        model=model,
        base_url=s.base_url or settings.llm.base_url,
        api_key=api_key,
        max_chars=s.max_chars,
        timeout=s.timeout,
    )
    cache_dir = settings.ingest.data_dir.parent / "captions"
    return CachedCaptioner(inner, cache_dir=cache_dir)

def build_ingest_pipeline(embeddings: Embeddings | None = None,
                          store: VectorStore | None = None,
                          settings: Settings | None = None,
                          *, prefer_memory: bool = False,
                          kb: str | None = None) -> IngestPipeline:
    """`kb` 给这批数据打知识库标签，检索时可用 kb_name 过滤（多知识库隔离）。"""
    settings = settings or get_settings()
    embeddings = embeddings or build_embeddings()
    store = store or build_store(settings, prefer_memory=prefer_memory)

    from ragkit.multimodal.image_store import ImageStore
    from ragkit.multimodal.pdf_loader import PdfLoader

    ocr = build_ocr_client(settings)
    image_store = ImageStore(root=settings.ingest.data_dir.parent)
    pdf_loader = PdfLoader(
        ocr, image_store=image_store, captioner=build_captioner(settings)
    )

    return IngestPipeline(
        store=store, embeddings=embeddings,
        chunk_size=settings.ingest.chunk_size, chunk_overlap=settings.ingest.chunk_overlap,
        min_chunk_chars=settings.ingest.min_chunk_chars, batch_size=settings.ingest.batch_size,
        kb=kb, pdf_loader=pdf_loader,
        image_root=settings.ingest.data_dir.parent,
    )

def answer(question: str, deps: Deps | None = None, *, session_id: str = "default",
           kb_name: str | None = None) -> dict:
    """最简问答入口（供 API 与脚本调用）。"""
    deps = deps or build_deps()
    state = run(question, deps, session_id=session_id, kb_name=kb_name)
    return {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "hits": [
            {"source": h.get("source"), "title": h.get("title"), "score": h.get("score"),
             "image_path": h.get("image_path", "")}
            for h in state.get("hits", [])[:10]
        ],
        "route": state.get("route", ""),
        "grounded": state.get("grounded", False),
        "confidence": state.get("confidence", 0.0),
        "timings_ms": state.get("timings", {}),
        "trace": state.get("trace", []),
    }
