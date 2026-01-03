from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequestModel(BaseModel):
    user_id: int | None = None
    seed_movie_id: int | None = None
    liked_movie_ids: list[int] = Field(default_factory=list)
    disliked_movie_ids: list[int] = Field(default_factory=list)
    query: str | None = None
    mood: str | None = None
    time_of_day: str | None = None
    top_k: int = 10
    genres: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None


class RecommendationResultModel(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    score_breakdown: dict[str, float]
    explanation: str
    avg_rating: float
    overview: str = ""
    poster_url: str = ""


class ChatRequestModel(BaseModel):
    query: str
    top_k: int = 10


class ChatResponseModel(BaseModel):
    parsed_query: dict[str, Any]
    resolved_query: str | None = None
    results: list[RecommendationResultModel]


class FeedbackRequestModel(BaseModel):
    user_id: int | None = None
    movie_id: int
    sentiment: str
    source: str = "ui"
