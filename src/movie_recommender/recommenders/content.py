from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_recommender.features.content import (
    ContentFeatureStore,
    cosine_scores_from_profile,
    cosine_scores_from_text_query,
    pair_similarity,
    profile_embedding,
)


@dataclass
class ContentRecommender:
    store: ContentFeatureStore
    movies: pd.DataFrame

    def similar_movies(self, movie_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        if movie_id not in self.store.movie_id_to_index:
            return []
        profile = self.store.vector_for_movie(movie_id)
        scores = cosine_scores_from_profile(self.store, profile, candidate_movie_ids=self.store.movie_ids)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(candidate_id, score) for candidate_id, score in ranked if candidate_id != movie_id][:top_k]

    def score_candidates(
        self,
        seed_movie_id: int | None = None,
        liked_movie_ids: list[int] | None = None,
        disliked_movie_ids: list[int] | None = None,
        candidate_movie_ids: list[int] | None = None,
        query_text: str | None = None,
    ) -> dict[int, float]:
        liked_movie_ids = liked_movie_ids or []
        disliked_movie_ids = disliked_movie_ids or []
        candidate_movie_ids = candidate_movie_ids or self.store.movie_ids

        profile_parts: list[np.ndarray] = []
        if seed_movie_id is not None and seed_movie_id in self.store.movie_id_to_index:
            profile_parts.append(self.store.vector_for_movie(seed_movie_id))
        if liked_movie_ids:
            profile_parts.append(profile_embedding(self.store, liked_movie_ids))
        if disliked_movie_ids:
            profile_parts.append(-profile_embedding(self.store, disliked_movie_ids))

        profile_scores: dict[int, float] = {}
        if profile_parts:
            profile = np.mean(np.vstack(profile_parts), axis=0)
            profile_scores = cosine_scores_from_profile(self.store, profile, candidate_movie_ids=candidate_movie_ids)

        query_scores = cosine_scores_from_text_query(self.store, query_text or "", candidate_movie_ids=candidate_movie_ids)

        if profile_scores and query_scores:
            profile_weight = 0.75 if seed_movie_id is not None or liked_movie_ids else 0.55
            query_weight = 1.0 - profile_weight
            return {
                movie_id: float(profile_weight * profile_scores.get(movie_id, 0.0) + query_weight * query_scores.get(movie_id, 0.0))
                for movie_id in candidate_movie_ids
            }

        if profile_scores:
            return {movie_id: float(profile_scores.get(movie_id, 0.0)) for movie_id in candidate_movie_ids}

        if query_scores:
            return {movie_id: float(query_scores.get(movie_id, 0.0)) for movie_id in candidate_movie_ids}

        if not profile_parts and not query_scores:
            return {movie_id: 0.0 for movie_id in candidate_movie_ids}
        return {movie_id: 0.0 for movie_id in candidate_movie_ids}

    def pair_similarity(self, movie_id_a: int, movie_id_b: int) -> float:
        return pair_similarity(self.store, movie_id_a, movie_id_b)
