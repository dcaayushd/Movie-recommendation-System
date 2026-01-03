from __future__ import annotations

import pandas as pd

from movie_recommender.services.evaluation import evaluate_rankings


def test_training_produces_models_and_metrics(trained_bundle):
    assert trained_bundle.hybrid_weights
    assert "audience" in trained_bundle.evaluation_report
    assert "svd" in trained_bundle.evaluation_report
    assert "matrix_factorization" in trained_bundle.evaluation_report
    assert "autoencoder" in trained_bundle.evaluation_report
    assert "hybrid" in trained_bundle.evaluation_report
    assert trained_bundle.content_features.movie_ids


def test_content_similarity_returns_neighbors(trained_bundle):
    similar = trained_bundle.content_recommender.similar_movies(1, top_k=3)
    assert similar
    assert all(movie_id != 1 for movie_id, _score in similar)


def test_evaluate_rankings_can_use_restricted_candidate_pool():
    train = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "rating": 4.0, "timestamp": 1},
            {"user_id": 1, "movie_id": 2, "rating": 5.0, "timestamp": 2},
        ]
    )
    holdout = pd.DataFrame([{"user_id": 1, "movie_id": 3, "rating": 4.5, "timestamp": 3}])
    movies = pd.DataFrame({"movie_id": [1, 2, 3, 100000123]})
    seen_candidate_ids: list[list[int]] = []

    def scorer(_user_id: int, candidate_ids: list[int]) -> dict[int, float]:
        seen_candidate_ids.append(candidate_ids)
        return {movie_id: float(movie_id) for movie_id in candidate_ids}

    metrics = evaluate_rankings(
        scorer,
        train,
        holdout,
        movies,
        top_k=3,
        candidate_movie_ids=[1, 2, 3],
    )

    assert metrics["precision@3"] >= 0.0
    assert seen_candidate_ids == [[3]]
