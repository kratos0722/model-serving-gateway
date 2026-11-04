"""FastAPI inference gateway.

Owns the flow from a versioned model artifact to an online prediction:
validate -> route (canary / A/B) -> score (timeout-guarded) -> shadow compare
-> record metrics -> sanitized response.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import GatewayConfig
from .metrics import Metrics
from .model import FeatureError, LogisticModel
from .registry import ModelRegistry, UnknownModelError
from .routing import Router
from .schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)


class ScorerTimeout(Exception):
    """Raised when scoring exceeds the configured per-request timeout."""


def _score(model: LogisticModel, features, timeout_s: float, executor):
    """Run the (synchronous) scorer under a hard timeout."""
    future = executor.submit(model.predict, features)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise ScorerTimeout() from exc


def create_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    config = config or GatewayConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.registry = ModelRegistry.from_dir(config.models_dir)
        app.state.router = Router(config)
        app.state.metrics = Metrics(window_size=config.latency_window)
        app.state.executor = ThreadPoolExecutor(max_workers=8)
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Model Serving Gateway",
        version="0.1.0",
        description="Versioned online inference with canary, shadow and A/B routing.",
        lifespan=lifespan,
    )

    # ---- error handlers (never leak stack traces) -----------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": "Invalid request body."},
        )

    @app.exception_handler(UnknownModelError)
    async def _unknown_model_handler(request: Request, exc: UnknownModelError):
        return JSONResponse(
            status_code=404,
            content={
                "error": "unknown_model",
                "detail": "Requested model version is not available.",
            },
        )

    @app.exception_handler(FeatureError)
    async def _feature_handler(request: Request, exc: FeatureError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_features",
                "detail": "Request features do not satisfy the model schema.",
            },
        )

    @app.exception_handler(ScorerTimeout)
    async def _timeout_handler(request: Request, exc: ScorerTimeout):
        return JSONResponse(
            status_code=503,
            content={"error": "scorer_timeout", "detail": "Scoring timed out."},
        )

    @app.exception_handler(Exception)
    async def _catch_all(request: Request, exc: Exception):
        # Sanitized: no internal detail or traceback ever reaches the client.
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An internal error occurred."},
        )

    # ---- routes ----------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return HealthResponse(
            status="ok", models_loaded=len(app.state.registry.list_models())
        )

    @app.get("/models", response_model=list[ModelInfo])
    async def models():
        return app.state.registry.list_models()

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            app.state.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        cfg: GatewayConfig = app.state.config
        registry: ModelRegistry = app.state.registry
        router: Router = app.state.router
        metrics: Metrics = app.state.metrics

        routing_key = req.user_id or uuid4().hex

        # Choose primary version: explicit override, else configured strategy.
        variant: Optional[str] = None
        if req.version is not None:
            primary_version = req.version
        else:
            primary_version, variant = router.route(routing_key, req.user_id)

        # Raises UnknownModelError -> sanitized 404.
        primary_model = registry.get(cfg.model_name, primary_version)

        start = time.perf_counter()
        prob, label = _score(
            primary_model, req.features, cfg.request_timeout_s, app.state.executor
        )
        elapsed = time.perf_counter() - start

        metrics.observe_latency(primary_version, elapsed)
        metrics.record_request(
            primary_version, "positive" if label == 1 else "negative"
        )

        # Shadow mode: score the shadow version in-line, record divergence, but
        # NEVER let it affect the served response.
        shadowed = False
        shadow_version = cfg.shadow_version or None
        if shadow_version and shadow_version != primary_version:
            if (cfg.model_name, shadow_version) in registry:
                shadow_model = registry.get(cfg.model_name, shadow_version)
                try:
                    shadow_prob, _ = _score(
                        shadow_model,
                        req.features,
                        cfg.request_timeout_s,
                        app.state.executor,
                    )
                    diverged = abs(shadow_prob - prob) > cfg.shadow_threshold
                    metrics.record_shadow(primary_version, shadow_version, diverged)
                    shadowed = True
                except (ScorerTimeout, FeatureError):
                    # A failing shadow must not break the primary response.
                    metrics.record_shadow(primary_version, shadow_version, True)
                    shadowed = True

        return PredictResponse(
            model_name=cfg.model_name,
            version_served=primary_version,
            variant=variant,
            probability=prob,
            label=label,
            shadowed=shadowed,
            shadow_version=shadow_version if shadowed else None,
        )

    return app


app = create_app()
