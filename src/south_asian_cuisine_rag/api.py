from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .errors import ArtifactError, QueryValidationError, RagError
from .logging_config import configure_logging
from .service import RagService

LOGGER = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceResponse(BaseModel):
    source_id: str
    title: str
    section: str
    publisher: str
    url: str
    excerpt: str
    similarity_score: float
    rerank_score: float | None
    ranking_score: float | None


class TimingResponse(BaseModel):
    retrieval: float
    generation: float


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool
    model: str
    sources: list[SourceResponse]
    timings_ms: TimingResponse


def create_app(
    settings: Settings | None = None, *, service: RagService | None = None
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    configure_logging(active_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = active_settings
        app.state.inference_semaphore = asyncio.Semaphore(
            active_settings.request_concurrency
        )
        app.state.startup_error = None
        if service is not None:
            app.state.rag_service = service
        else:
            try:
                app.state.rag_service = RagService.from_settings(active_settings)
                LOGGER.info("RAG artifacts loaded; models will load lazily")
            except ArtifactError as exc:
                app.state.rag_service = None
                app.state.startup_error = str(exc)
                LOGGER.error("RAG service is not ready: %s", exc)
        yield

    app = FastAPI(
        title="South Asian Cuisine RAG API",
        version="2.0.0",
        docs_url="/docs" if active_settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "Unhandled request failure",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    @app.exception_handler(QueryValidationError)
    async def query_error_handler(_: Request, exc: QueryValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RagError)
    async def rag_error_handler(_: Request, exc: RagError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = active_settings.api_key
        if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid X-API-Key header is required",
            )

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def readiness(request: Request) -> dict[str, object]:
        if request.app.state.rag_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=request.app.state.startup_error or "RAG service is not initialized",
            )
        store = request.app.state.rag_service.retriever.store
        return {
            "status": "ready",
            "model": active_settings.generation_model,
            "artifact_created_at": store.manifest.created_at,
            "chunks": store.manifest.chunk_count,
        }

    @app.post(
        "/v1/answers",
        response_model=AnswerResponse,
        tags=["answers"],
    )
    async def answer(
        payload: AnswerRequest,
        request: Request,
        _: None = Depends(require_api_key),
    ) -> dict[str, object]:
        rag_service: RagService | None = request.app.state.rag_service
        if rag_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=request.app.state.startup_error or "RAG service is not ready",
            )
        async with request.app.state.inference_semaphore:
            result = await run_in_threadpool(
                partial(rag_service.answer, payload.question, top_k=payload.top_k)
            )
        return result.to_public_dict()

    return app


app = create_app()
