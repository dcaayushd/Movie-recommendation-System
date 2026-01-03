from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from movie_recommender.api.schemas import (
    ChatRequestModel,
    ChatResponseModel,
    FeedbackRequestModel,
    RecommendationRequestModel,
    RecommendationResultModel,
)
from movie_recommender.config.settings import Settings, get_settings
from movie_recommender.llm.factory import build_llm_backend
from movie_recommender.services.inference import MovieRecommenderService, RecommendationRequest
from movie_recommender.services.metrics import RequestMetrics


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    metrics = RequestMetrics()
    service_holder: dict[str, MovieRecommenderService | None] = {"service": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.bundle_path.exists():
            llm_backend = build_llm_backend(settings)
            service_holder["service"] = MovieRecommenderService.from_path(settings, llm_backend=llm_backend)
        yield

    app = FastAPI(title="Movie Recommendation System", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def collect_metrics(request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start_time
        metrics.record(request.url.path, request.method, response.status_code, latency)
        return response

    def _service() -> MovieRecommenderService:
        service = service_holder["service"]
        if service is None:
            raise HTTPException(status_code=503, detail="Model bundle not found. Run training first.")
        return service

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "bundle_ready": service_holder["service"] is not None}

    @app.get("/metrics")
    async def get_metrics() -> dict:
        return metrics.route_summary()

    @app.get("/status")
    async def status() -> dict:
        service = service_holder["service"]
        runtime = {
            "bundle_ready": service is not None,
            "server_started_at": metrics.started_at,
            "metrics": metrics.route_summary(),
            "recent_requests": metrics.recent_activity(),
        }
        if service is not None:
            runtime["bundle"] = {
                "dataset_name": service.bundle.dataset_manifest.get("dataset_name"),
                "movies": int(len(service.bundle.movies)),
                "ratings": int(len(service.bundle.ratings)),
                "users": int(len(service.bundle.users)),
                "hybrid_weights": service.bundle.hybrid_weights,
                "llm_backend": service.llm_backend.__class__.__name__,
            }
        return runtime

    @app.post("/recommend", response_model=list[RecommendationResultModel])
    async def recommend(payload: RecommendationRequestModel) -> list[RecommendationResultModel]:
        request = RecommendationRequest(**payload.model_dump())
        results = _service().recommend(request)
        return [RecommendationResultModel(**result.__dict__) for result in results]

    @app.get("/similar/{movie_id}", response_model=list[RecommendationResultModel])
    async def similar(movie_id: int, top_k: int = 10) -> list[RecommendationResultModel]:
        results = _service().similar_movies(movie_id, top_k=top_k)
        return [RecommendationResultModel(**result.__dict__) for result in results]

    @app.post("/chat", response_model=ChatResponseModel)
    async def chat(payload: ChatRequestModel) -> ChatResponseModel:
        response = _service().chat(payload.query, top_k=payload.top_k)
        return ChatResponseModel(
            parsed_query=response["parsed_query"],
            resolved_query=response.get("resolved_query"),
            results=[RecommendationResultModel(**result.__dict__) for result in response["results"]],
        )

    @app.post("/feedback")
    async def feedback(payload: FeedbackRequestModel) -> dict:
        _service().record_feedback(payload.model_dump())
        return {"status": "recorded"}

    return app


app = create_app()
