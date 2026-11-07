"""HTTP 服务（FastAPI）。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ragkit.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchImageRequest,
    SearchRequest,
    SearchResponse,
)
from ragkit.config import get_settings, redacted
from ragkit.logging_setup import get_logger

log = get_logger("api")

_STATE: dict[str, Any] = {}

def get_deps():
    """取全局依赖（生命周期内复用）。"""
    if "deps" not in _STATE:
        from ragkit.bootstrap import build_deps

        _STATE["deps"] = build_deps()
    return _STATE["deps"]

async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """可选鉴权：没配 key 就放行（本地开发），配了就必须带。"""
    expected = get_settings().api.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时装配一次依赖并缓存；关闭时清理。"""
    settings = get_settings()
    from ragkit.bootstrap import build_deps

    log.info("启动服务: env=%s", settings.env)
    _STATE["deps"] = build_deps()
    _STATE["settings"] = settings
    yield
    _STATE.clear()
    log.info("服务已停止")

def create_app() -> FastAPI:
    """构造 FastAPI 应用：CORS、图片静态路由、四个接口。"""
    settings = get_settings()
    app = FastAPI(
        title="ragkit · 企业级 RAG 服务",
        version="1.0.0",
        description="混合检索（稠密 + BM25）+ RRF 融合 + Corrective RAG 编排",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.api.cors_origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from fastapi.staticfiles import StaticFiles

    images_dir = settings.ingest.data_dir.parent / "images"
    app.mount("/images", StaticFiles(directory=str(images_dir), check_dir=False),
              name="images")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """健康检查：库类型 + 文档数 + 脱敏配置（上线自检首选）。"""
        deps = get_deps()
        try:
            count = deps.service.store.count()
        except Exception as exc:  # noqa: BLE001
            log.warning("统计文档数失败: %s", exc)
            count = -1
        return HealthResponse(
            status="ok", env=get_settings().env,
            store=type(deps.service.store).__name__, doc_count=count,
            config=redacted(),
        )

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    def chat(req: ChatRequest) -> ChatResponse:
        """完整问答入口：走图（路由→wiki→检索→生成→校验）。"""
        from ragkit.bootstrap import answer

        try:
            result = answer(req.question, deps=get_deps(),
                            session_id=req.session_id, kb_name=req.kb_name,
                            groups=req.groups)
        except Exception as exc:  # noqa: BLE001
            log.exception("/chat 失败: %s", exc)
            raise HTTPException(status_code=503, detail=f"问答服务异常: {exc}") from exc
        return ChatResponse(**result)

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
    def search(req: SearchRequest) -> SearchResponse:
        """只检索不生成：返回 top-k 命中及两路分数。"""
        ctx = get_deps().service.search(req.question, top_k=req.top_k, acl=req.groups)
        return SearchResponse(
            query=ctx.query,
            hits=[
                {"chunk_id": h.chunk_id, "text": h.text[:500], "score": h.score,
                 "source": h.source, "title": h.title,
                 "image_path": h.image_path,
                 "dense_score": h.dense_score, "sparse_score": h.sparse_score}
                for h in ctx.hits
            ],
            elapsed_ms=ctx.elapsed_ms,
        )

    @app.post("/search-image", response_model=SearchResponse,
              dependencies=[Depends(require_api_key)])
    def search_image(req: SearchImageRequest) -> SearchResponse:
        """图搜图：给一张图，返回相关的图片块（及其 OCR 文字）。"""
        import base64
        import binascii

        try:
            image_bytes = base64.b64decode(req.image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=422, detail="image_base64 不是合法的 base64") from None
        ctx = get_deps().service.search_image(image_bytes, top_k=req.top_k)
        return SearchResponse(
            query="<image>",
            hits=[
                {"chunk_id": h.chunk_id, "text": h.text[:500], "score": h.score,
                 "source": h.source, "title": h.title,
                 "image_path": h.image_path,
                 "dense_score": h.dense_score, "sparse_score": h.sparse_score}
                for h in ctx.hits
            ],
            elapsed_ms=ctx.elapsed_ms,
        )

    @app.post("/admin/ingest", response_model=IngestResponse,
              dependencies=[Depends(require_api_key)])
    def ingest(req: IngestRequest) -> IngestResponse:
        from pathlib import Path

        from ragkit.bootstrap import build_ingest_pipeline

        pipeline = build_ingest_pipeline()
        stats = pipeline.run(Path(req.data_dir), recreate=req.recreate)
        return IngestResponse(**stats.to_dict())

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    s = get_settings().api
    uvicorn.run("ragkit.api.server:app", host=s.host, port=s.port, reload=False)
