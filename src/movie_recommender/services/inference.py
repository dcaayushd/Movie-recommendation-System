from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from movie_recommender.config.settings import Settings
from movie_recommender.features.content import ensure_query_content_features
from movie_recommender.llm.base import LLMBackend, NoOpLLMBackend, fallback_explanation
from movie_recommender.ranking.audience import AudienceRecommender, enrich_movie_audience_signals
from movie_recommender.recommenders.hybrid import HybridRanker
from movie_recommender.services.bundle import ModelBundle
from movie_recommender.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    user_id: int | None = None
    seed_movie_id: int | None = None
    liked_movie_ids: list[int] = field(default_factory=list)
    disliked_movie_ids: list[int] = field(default_factory=list)
    query: str | None = None
    mood: str | None = None
    time_of_day: str | None = None
    top_k: int = 10
    genres: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None


@dataclass
class RecommendationResult:
    movie_id: int
    title: str
    genres: str
    score: float
    score_breakdown: dict[str, float]
    explanation: str
    avg_rating: float
    overview: str = ""
    poster_url: str = ""


class MovieRecommenderService:
    def __init__(self, bundle: ModelBundle, llm_backend: LLMBackend | None = None, feedback_log_path: Path | None = None) -> None:
        self.bundle = bundle
        self.llm_backend = llm_backend or NoOpLLMBackend()
        self.feedback_log_path = feedback_log_path
        self.movies = enrich_movie_audience_signals(bundle.movies.copy(), ratings=bundle.ratings)
        self.bundle.content_features = ensure_query_content_features(bundle.content_features, self.movies)
        self.bundle.content_recommender.store = self.bundle.content_features
        self.bundle.content_recommender.movies = self.movies
        self.movie_lookup = self.movies.set_index("movie_id")
        self.movie_title_lookup = {str(row.clean_title).lower(): int(row.movie_id) for row in self.movies.itertuples(index=False)}
        self.ratings_by_user = bundle.ratings.groupby("user_id")["movie_id"].agg(set).to_dict()
        self.hybrid_ranker = HybridRanker(bundle.hybrid_weights)
        self.audience_recommender = AudienceRecommender.from_movies(self.movies)
        self.global_popular_titles = self.movies.sort_values(["rating_count", "avg_rating"], ascending=False)["clean_title"].head(5).tolist()
        self.max_rating_count = float(self.movies["rating_count"].max()) if "rating_count" in self.movies.columns and not self.movies.empty else 1.0
        normalized_ratings = self.movies["avg_rating"].fillna(0.0).map(self._normalize_rating_value) if "avg_rating" in self.movies.columns else pd.Series(dtype=float)
        rating_counts = self.movies["rating_count"].fillna(0.0).astype(float) if "rating_count" in self.movies.columns else pd.Series(dtype=float)
        self.global_rating_baseline = float(normalized_ratings.mean()) if not normalized_ratings.empty else 0.0
        self.quality_vote_floor = float(rating_counts.quantile(0.75)) if not rating_counts.empty else 1.0
        if self.quality_vote_floor <= 0:
            self.quality_vote_floor = 1.0

    @classmethod
    def from_path(cls, settings: Settings, llm_backend: LLMBackend | None = None) -> "MovieRecommenderService":
        bundle = ModelBundle.load(settings.bundle_path)
        return cls(bundle=bundle, llm_backend=llm_backend, feedback_log_path=settings.feedback_log_path)

    def resolve_movie_id(self, title: str | None) -> int | None:
        if not title:
            return None
        lowered = title.lower().strip()
        if lowered in self.movie_title_lookup:
            return self.movie_title_lookup[lowered]
        matches = get_close_matches(lowered, list(self.movie_title_lookup.keys()), n=1, cutoff=0.65)
        if matches:
            return self.movie_title_lookup[matches[0]]
        return None

    def similar_movies(self, movie_id: int, top_k: int = 10) -> list[RecommendationResult]:
        candidates = self.bundle.content_recommender.similar_movies(movie_id, top_k=top_k)
        return [self._result_from_movie_id(candidate_id, score, {"content": score}, ["it is content-similar to your selected movie"]) for candidate_id, score in candidates]

    def chat(self, query: str, top_k: int = 10) -> dict:
        initial_parsed = self.llm_backend.parse_query(query)
        seed_movie_id = self.resolve_movie_id(initial_parsed.get("seed_movie_title")) if initial_parsed.get("seed_movie_title") else None
        request = RecommendationRequest(
            seed_movie_id=seed_movie_id,
            query=query,
            mood=initial_parsed.get("mood"),
            time_of_day=initial_parsed.get("time_of_day"),
            top_k=initial_parsed.get("top_k") or top_k,
            genres=initial_parsed.get("genres") or [],
        )
        request = self._prepare_request(request)
        parsed = self.llm_backend.parse_query(request.query) if request.query else {}
        results = self.recommend(request)
        return {"parsed_query": parsed, "resolved_query": request.query, "results": results}

    def recommend(self, request: RecommendationRequest) -> list[RecommendationResult]:
        request = self._prepare_request(request)
        parsed_query = self.llm_backend.parse_query(request.query) if request.query else {}
        if request.seed_movie_id is None:
            request.seed_movie_id = self.resolve_movie_id(parsed_query.get("seed_movie_title")) if parsed_query.get("seed_movie_title") else None
        request.genres = request.genres or parsed_query.get("genres") or []
        request.mood = request.mood or parsed_query.get("mood")
        request.time_of_day = request.time_of_day or parsed_query.get("time_of_day")
        request.top_k = request.top_k or parsed_query.get("top_k") or 10

        candidate_ids = self._candidate_movie_ids(request)
        component_scores = self._component_scores(request, candidate_ids)
        ranker = HybridRanker(self._dynamic_weights(request))
        combined_scores, breakdown = ranker.combine(component_scores)
        reranked = self._session_rerank(request, combined_scores, breakdown)
        results = [
            self._result_from_movie_id(movie_id, score, breakdown[movie_id], self._reasons_for_movie(request, movie_id, breakdown[movie_id]))
            for movie_id, score in reranked[: request.top_k]
        ]
        return results

    def _prepare_request(self, request: RecommendationRequest) -> RecommendationRequest:
        if not self._needs_auto_prompt(request):
            return request
        auto_query = self._auto_prompt(request)
        return RecommendationRequest(
            user_id=request.user_id,
            seed_movie_id=request.seed_movie_id,
            liked_movie_ids=list(request.liked_movie_ids),
            disliked_movie_ids=list(request.disliked_movie_ids),
            query=auto_query,
            mood=request.mood,
            time_of_day=request.time_of_day,
            top_k=request.top_k,
            genres=list(request.genres),
            year_from=request.year_from,
            year_to=request.year_to,
        )

    def _needs_auto_prompt(self, request: RecommendationRequest) -> bool:
        generic_queries = {
            "",
            "recommend like this",
            "recommend me like this",
            "recommend on the basis of this",
            "recommend me on the basis of this",
            "recommend me",
        }
        query = (request.query or "").strip().lower()
        if not query:
            return True
        return query in generic_queries

    def _auto_prompt(self, request: RecommendationRequest) -> str:
        parts = ["Recommend movies"]
        if request.seed_movie_id is not None and request.seed_movie_id in self.movie_lookup.index:
            seed_title = self.movie_lookup.loc[request.seed_movie_id]["clean_title"]
            parts.append(f"similar to {seed_title}")
        if request.genres:
            parts.append(f"with genres {', '.join(request.genres)}")
        if request.mood:
            parts.append(f"for a {request.mood} mood")
        if request.time_of_day:
            parts.append(f"for {request.time_of_day}")
        if request.user_id is None and not request.seed_movie_id and not request.genres:
            parts.append(f"like {', '.join(self.global_popular_titles[:3])}")
        return " ".join(parts)

    def _dynamic_weights(self, request: RecommendationRequest) -> dict[str, float]:
        weights = dict(self.bundle.hybrid_weights)
        weights.setdefault("popularity", 0.08)
        weights.setdefault("content", 0.26)
        weights.setdefault("audience", 0.22)
        weights.setdefault("svd", 0.16)
        weights.setdefault("matrix_factorization", 0.18)
        weights.setdefault("autoencoder", 0.10)
        has_seed = request.seed_movie_id is not None
        has_context = bool(request.genres or request.mood or request.time_of_day)
        has_session_feedback = bool(request.liked_movie_ids or request.disliked_movie_ids)
        has_profile = request.user_id in self.bundle.user_profiles if request.user_id is not None else False
        has_query = bool((request.query or "").strip())

        if has_seed:
            weights["content"] = weights.get("content", 0.0) + 0.16
            weights["audience"] = weights.get("audience", 0.0) + 0.14
            weights["popularity"] = max(0.02, weights.get("popularity", 0.0) - 0.10)
        if has_query:
            weights["content"] = weights.get("content", 0.0) + 0.10
            weights["audience"] = weights.get("audience", 0.0) + 0.16
            weights["popularity"] = max(0.02, weights.get("popularity", 0.0) - 0.06)
        if has_context:
            weights["content"] = weights.get("content", 0.0) + 0.08
            weights["audience"] = weights.get("audience", 0.0) + 0.06
            weights["svd"] = max(0.05, weights.get("svd", 0.0) - 0.03)
        if has_session_feedback:
            weights["content"] = weights.get("content", 0.0) + 0.05
            weights["audience"] = weights.get("audience", 0.0) + 0.05
            weights["matrix_factorization"] = weights.get("matrix_factorization", 0.0) + 0.04
        if not has_profile and not has_session_feedback:
            weights["popularity"] = weights.get("popularity", 0.0) + 0.02
            weights["content"] = weights.get("content", 0.0) + 0.05
            weights["audience"] = weights.get("audience", 0.0) + 0.09
            weights["svd"] = max(0.03, weights.get("svd", 0.0) - 0.04)
            weights["matrix_factorization"] = max(0.03, weights.get("matrix_factorization", 0.0) - 0.04)
            weights["autoencoder"] = max(0.03, weights.get("autoencoder", 0.0) - 0.04)

        total = sum(max(value, 0.0) for value in weights.values()) or 1.0
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def record_feedback(self, payload: dict) -> None:
        if self.feedback_log_path is None:
            return
        ensure_dir(self.feedback_log_path.parent)
        with self.feedback_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _candidate_movie_ids(self, request: RecommendationRequest) -> list[int]:
        movies = self.movies.copy()
        filtered_movies = movies
        if request.genres:
            lowered_filters = {genre.lower() for genre in request.genres}
            filtered_movies = filtered_movies[
                filtered_movies["genres"].str.lower().apply(lambda genres: bool(lowered_filters & set(genres.replace("|", " ").split())))
            ]
        if request.year_from is not None:
            filtered_movies = filtered_movies[filtered_movies["year"] >= request.year_from]
        if request.year_to is not None:
            filtered_movies = filtered_movies[filtered_movies["year"] <= request.year_to]

        if filtered_movies.empty:
            filtered_movies = movies

        candidate_ids = filtered_movies["movie_id"].astype(int).tolist()
        seen = set(self.ratings_by_user.get(request.user_id, set())) if request.user_id is not None else set()
        excluded = seen | set(request.disliked_movie_ids or [])
        if request.seed_movie_id is not None:
            excluded.add(request.seed_movie_id)
        return [movie_id for movie_id in candidate_ids if movie_id not in excluded]

    def _component_scores(self, request: RecommendationRequest, candidate_ids: list[int]) -> dict[str, dict[int, float]]:
        user_profile = self.bundle.user_profiles.get(request.user_id or -1, {})
        liked_movie_ids = list(dict.fromkeys((user_profile.get("liked_movie_ids", []) + request.liked_movie_ids)))
        disliked_movie_ids = list(dict.fromkeys((user_profile.get("disliked_movie_ids", []) + request.disliked_movie_ids)))

        return {
            "popularity": self.bundle.popularity.score_movies(candidate_ids),
            "content": self.bundle.content_recommender.score_candidates(
                seed_movie_id=request.seed_movie_id,
                liked_movie_ids=liked_movie_ids,
                disliked_movie_ids=disliked_movie_ids,
                candidate_movie_ids=candidate_ids,
                query_text=request.query,
            ),
            "audience": self.audience_recommender.score_candidates(
                seed_movie_id=request.seed_movie_id,
                liked_movie_ids=liked_movie_ids,
                disliked_movie_ids=disliked_movie_ids,
                candidate_movie_ids=candidate_ids,
                query_text=request.query,
            ),
            "svd": self.bundle.svd_model.score_user(request.user_id or -1, candidate_ids),
            "matrix_factorization": self.bundle.matrix_factorization_model.score_user(request.user_id or -1, candidate_ids),
            "autoencoder": self.bundle.autoencoder_model.score_user(request.user_id or -1, candidate_ids),
        }

    def _session_rerank(
        self,
        request: RecommendationRequest,
        combined_scores: dict[int, float],
        breakdown: dict[int, dict[str, float]],
    ) -> list[tuple[int, float]]:
        candidate_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
        candidate_limit = min(len(candidate_ids), max(90, request.top_k * 12))
        candidate_ids = candidate_ids[:candidate_limit]
        selection_limit = min(len(candidate_ids), max(request.top_k, request.top_k * 2))
        liked_movie_ids = request.liked_movie_ids or []
        disliked_movie_ids = request.disliked_movie_ids or []

        session_profile = None
        if liked_movie_ids or disliked_movie_ids or request.seed_movie_id is not None:
            session_profile_scores = self.bundle.content_recommender.score_candidates(
                seed_movie_id=request.seed_movie_id,
                liked_movie_ids=liked_movie_ids,
                disliked_movie_ids=disliked_movie_ids,
                candidate_movie_ids=candidate_ids,
            )
            session_profile = session_profile_scores

        if not candidate_ids:
            return []

        base_scores = np.array([combined_scores[movie_id] for movie_id in candidate_ids], dtype=float)
        quality_bonus = np.array([0.1 * self._quality_score(movie_id) for movie_id in candidate_ids], dtype=float)
        session_bonus = (
            np.array([0.2 * session_profile.get(movie_id, 0.0) for movie_id in candidate_ids], dtype=float)
            if session_profile is not None
            else np.zeros(len(candidate_ids), dtype=float)
        )
        mood_bonus = (
            np.array([0.05 * self._mood_score(request.mood, movie_id) for movie_id in candidate_ids], dtype=float)
            if request.mood
            else np.zeros(len(candidate_ids), dtype=float)
        )
        time_bonus = (
            np.array([0.03 * self._time_score(request.time_of_day, movie_id) for movie_id in candidate_ids], dtype=float)
            if request.time_of_day
            else np.zeros(len(candidate_ids), dtype=float)
        )
        adjusted_base = base_scores + quality_bonus + session_bonus + mood_bonus + time_bonus
        diversity_matrix = self._candidate_similarity_matrix(candidate_ids)
        max_similarity = np.zeros(len(candidate_ids), dtype=float)

        reranked: list[tuple[int, float]] = []
        remaining_indices = np.arange(len(candidate_ids), dtype=int)
        has_selected = False

        while remaining_indices.size and len(reranked) < selection_limit:
            diversity_penalty = -0.1 * max_similarity[remaining_indices] if has_selected else np.zeros(remaining_indices.size, dtype=float)
            candidate_scores = adjusted_base[remaining_indices] + diversity_penalty
            best_position = int(np.argmax(candidate_scores))
            best_index = int(remaining_indices[best_position])
            best_movie_id = candidate_ids[best_index]
            best_score = float(candidate_scores[best_position])

            adjustments = {
                "quality": float(quality_bonus[best_index]),
                "session": float(session_bonus[best_index]),
                "mood": float(mood_bonus[best_index]),
                "time": float(time_bonus[best_index]),
                "diversity": float(-0.1 * max_similarity[best_index]) if has_selected else 0.0,
            }
            breakdown.setdefault(best_movie_id, {})
            for name, value in adjustments.items():
                if abs(value) > 1e-9:
                    breakdown[best_movie_id][name] = float(value)
            breakdown[best_movie_id]["rerank"] = best_score - combined_scores[best_movie_id]
            reranked.append((best_movie_id, best_score))

            if diversity_matrix.size:
                max_similarity = np.maximum(max_similarity, diversity_matrix[best_index])
            has_selected = True
            remaining_indices = remaining_indices[remaining_indices != best_index]

        return reranked

    def _candidate_similarity_matrix(self, candidate_ids: list[int]) -> np.ndarray:
        if not candidate_ids:
            return np.zeros((0, 0), dtype=float)
        store = self.bundle.content_features
        indices = store.indices_for_movies(candidate_ids)
        if not indices:
            return np.zeros((0, 0), dtype=float)
        if store.semantic_embeddings is not None:
            candidate_vectors = store.semantic_embeddings[indices]
            return np.clip(candidate_vectors @ candidate_vectors.T, -1.0, 1.0)
        candidate_matrix = store.tfidf_matrix[indices]
        return cosine_similarity(candidate_matrix, candidate_matrix)

    def _quality_score(self, movie_id: int) -> float:
        row = self.movie_lookup.loc[movie_id]
        avg_rating = float(row.get("avg_rating", 0.0))
        rating_count = float(row.get("rating_count", 0.0))
        rating_signal = self._normalize_rating_value(avg_rating)
        weighted_rating = (
            ((rating_count / (rating_count + self.quality_vote_floor)) * rating_signal)
            + ((self.quality_vote_floor / (rating_count + self.quality_vote_floor)) * self.global_rating_baseline)
            if (rating_count + self.quality_vote_floor) > 0
            else self.global_rating_baseline
        )
        count_denominator = np.log1p(self.max_rating_count) if self.max_rating_count > 0 else 1.0
        count_signal = float(np.log1p(rating_count) / count_denominator) if count_denominator else 0.0
        return 0.8 * weighted_rating + 0.2 * count_signal

    @staticmethod
    def _normalize_rating_value(avg_rating: float) -> float:
        rating_value = float(avg_rating or 0.0)
        rating_scale = 10.0 if rating_value > 5.0 else 5.0
        return min(max(rating_value / rating_scale, 0.0), 1.0)

    def _mood_score(self, mood: str, movie_id: int) -> float:
        row = self.movie_lookup.loc[movie_id]
        descriptor = f"{row['genres']} {row.get('movie_tag_text', '')} {row.get('overview', '')}".lower()
        mood_to_terms = {
            "uplifting": ["comedy", "family", "feel-good", "hopeful"],
            "dark": ["crime", "thriller", "noir", "dark"],
            "funny": ["comedy", "funny", "satire"],
            "romantic": ["romance", "love", "relationship"],
            "thoughtful": ["drama", "thought-provoking", "philosophy"],
            "intense": ["action", "thriller", "war"],
            "relaxed": ["animation", "family", "musical"],
            "family": ["children", "animation", "family"],
        }
        return float(any(term in descriptor for term in mood_to_terms.get(mood.lower(), [])))

    def _time_score(self, time_of_day: str, movie_id: int) -> float:
        row = self.movie_lookup.loc[movie_id]
        descriptor = f"{row['genres']} {row.get('movie_tag_text', '')}".lower()
        mapping = {
            "morning": ["animation", "comedy", "family"],
            "afternoon": ["adventure", "comedy", "drama"],
            "evening": ["drama", "romance", "mystery"],
            "night": ["thriller", "crime", "horror", "noir"],
        }
        return float(any(term in descriptor for term in mapping.get(time_of_day.lower(), [])))

    def _reasons_for_movie(self, request: RecommendationRequest, movie_id: int, score_breakdown: dict[str, float]) -> list[str]:
        row = self.movie_lookup.loc[movie_id]
        reasons = []
        if request.seed_movie_id is not None:
            seed_title = self.movie_lookup.loc[request.seed_movie_id]["clean_title"]
            reasons.append(f"it shares content traits with {seed_title}")
        elif request.query:
            reasons.append("it matches the kind of movie you asked for")
        if score_breakdown.get("audience", 0.0) > 0.12:
            reasons.append("audience reactions and review language line up with your request")
        if request.mood:
            reasons.append(f"it matches a {request.mood} mood")
        if request.genres:
            reasons.append(f"it fits your requested genres: {', '.join(request.genres)}")
        if (score_breakdown.get("matrix_factorization", 0.0) + score_breakdown.get("autoencoder", 0.0) + score_breakdown.get("svd", 0.0)) > 0:
            reasons.append("your historical rating behavior aligns with it")
        if self._quality_score(movie_id) >= 0.8:
            reasons.append("it has especially strong ratings from viewers")
        audience_terms = self.audience_recommender.highlight_terms(movie_id)
        if audience_terms:
            reasons.append(f"audience tags often describe it as {', '.join(audience_terms)}")
        if row.get("rating_count", 0) > 50:
            reasons.append("it is also well-liked by similar viewers")
        return reasons

    def _result_from_movie_id(
        self,
        movie_id: int,
        score: float,
        score_breakdown: dict[str, float],
        reasons: list[str],
    ) -> RecommendationResult:
        row = self.movie_lookup.loc[movie_id]
        title = str(row["clean_title"])
        explanation = self.llm_backend.explain_recommendation(title, reasons, score_breakdown)
        if not explanation:
            explanation = fallback_explanation(title, reasons, score_breakdown)
        return RecommendationResult(
            movie_id=int(movie_id),
            title=title,
            genres=str(row["genres"]),
            score=float(score),
            score_breakdown={key: round(float(value), 5) for key, value in score_breakdown.items()},
            explanation=explanation,
            avg_rating=float(round(self._normalize_rating_value(float(row.get("avg_rating", 0.0))) * 5.0, 2)),
            overview=str(row.get("overview", "")),
            poster_url=str(row.get("poster_url", "")),
        )
