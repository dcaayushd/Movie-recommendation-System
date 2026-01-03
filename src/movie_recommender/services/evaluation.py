from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movie_recommender.ranking.audience import AudienceRecommender
from movie_recommender.recommenders.hybrid import HybridRanker
from movie_recommender.services.bundle import ModelBundle

LOGGER = logging.getLogger(__name__)


@dataclass
class RatingsSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def chronological_split(ratings: pd.DataFrame) -> RatingsSplit:
    sorted_ratings = ratings.sort_values(["user_id", "timestamp"])
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _user_id, frame in sorted_ratings.groupby("user_id"):
        if len(frame) >= 3:
            train_parts.append(frame.iloc[:-2])
            valid_parts.append(frame.iloc[[-2]])
            test_parts.append(frame.iloc[[-1]])
        elif len(frame) == 2:
            train_parts.append(frame.iloc[[0]])
            test_parts.append(frame.iloc[[1]])
        else:
            train_parts.append(frame)

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=ratings.columns)
        return pd.concat(parts, ignore_index=True)

    return RatingsSplit(train=_concat(train_parts), valid=_concat(valid_parts), test=_concat(test_parts))


def rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return math.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))


def mae(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = 1.0 if relevant else 0.0
    return dcg / ideal if ideal else 0.0


def evaluate_explicit_model(model, ratings: pd.DataFrame) -> dict[str, float]:
    actual: list[float] = []
    predicted: list[float] = []
    for row in ratings.itertuples(index=False):
        actual.append(float(row.rating))
        predicted.append(float(model.predict(int(row.user_id), int(row.movie_id))))
    return {"rmse": round(rmse(actual, predicted), 5), "mae": round(mae(actual, predicted), 5)}


def evaluate_rankings(
    scorer,
    train_ratings: pd.DataFrame,
    holdout_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_k: int = 10,
    candidate_movie_ids: list[int] | None = None,
) -> dict[str, float]:
    movie_ids = candidate_movie_ids or movies["movie_id"].astype(int).tolist()
    seen_by_user = train_ratings.groupby("user_id")["movie_id"].agg(set).to_dict()
    user_ids = holdout_ratings["user_id"].astype(int).unique().tolist()
    candidate_pool_by_user = {user_id: [movie_id for movie_id in movie_ids if movie_id not in seen_by_user.get(user_id, set())] for user_id in user_ids}

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    for row in holdout_ratings.itertuples(index=False):
        user_id = int(row.user_id)
        relevant = {int(row.movie_id)}
        candidates = candidate_pool_by_user.get(user_id, movie_ids)
        if int(row.movie_id) not in candidates:
            candidates = candidates + [int(row.movie_id)]
        scored = scorer(user_id, candidates)
        ranked = [movie_id for movie_id, _score in sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        precision_scores.append(precision_at_k(ranked, relevant, top_k))
        recall_scores.append(recall_at_k(ranked, relevant, top_k))
        ndcg_scores.append(ndcg_at_k(ranked, relevant, top_k))

    return {
        f"precision@{top_k}": round(float(np.mean(precision_scores)) if precision_scores else 0.0, 5),
        f"recall@{top_k}": round(float(np.mean(recall_scores)) if recall_scores else 0.0, 5),
        f"ndcg@{top_k}": round(float(np.mean(ndcg_scores)) if ndcg_scores else 0.0, 5),
    }


def evaluate_bundle(bundle: ModelBundle, split: RatingsSplit, top_k: int = 10) -> dict[str, dict[str, float]]:
    content_profiles = bundle.user_profiles
    audience_recommender = AudienceRecommender.from_movies(bundle.movies)
    ranking_movie_ids = sorted(
        set(split.train["movie_id"].astype(int))
        | set(split.valid["movie_id"].astype(int))
        | set(split.test["movie_id"].astype(int))
    )
    LOGGER.info(
        "Evaluating ranking metrics on observed movie universe",
        extra={"context": {"ranking_movies": len(ranking_movie_ids), "total_movies": int(len(bundle.movies))}},
    )

    def content_scorer(user_id: int, candidate_ids: list[int]) -> dict[int, float]:
        profile = content_profiles.get(user_id, {})
        liked_movie_ids = profile.get("liked_movie_ids", [])
        disliked_movie_ids = profile.get("disliked_movie_ids", [])
        return bundle.content_recommender.score_candidates(
            liked_movie_ids=liked_movie_ids,
            disliked_movie_ids=disliked_movie_ids,
            candidate_movie_ids=candidate_ids,
        )

    def popularity_scorer(_user_id: int, candidate_ids: list[int]) -> dict[int, float]:
        return bundle.popularity.score_movies(candidate_ids)

    def audience_scorer(user_id: int, candidate_ids: list[int]) -> dict[int, float]:
        profile = content_profiles.get(user_id, {})
        liked_movie_ids = profile.get("liked_movie_ids", [])
        disliked_movie_ids = profile.get("disliked_movie_ids", [])
        return audience_recommender.score_candidates(
            liked_movie_ids=liked_movie_ids,
            disliked_movie_ids=disliked_movie_ids,
            candidate_movie_ids=candidate_ids,
        )

    def model_scorer(model):
        return lambda user_id, candidate_ids: model.score_user(user_id, candidate_ids)

    hybrid_ranker = HybridRanker(bundle.hybrid_weights)

    def hybrid_scorer(user_id: int, candidate_ids: list[int]) -> dict[int, float]:
        component_scores = {
            "popularity": popularity_scorer(user_id, candidate_ids),
            "content": content_scorer(user_id, candidate_ids),
            "audience": audience_scorer(user_id, candidate_ids),
            "svd": bundle.svd_model.score_user(user_id, candidate_ids),
            "matrix_factorization": bundle.matrix_factorization_model.score_user(user_id, candidate_ids),
            "autoencoder": bundle.autoencoder_model.score_user(user_id, candidate_ids),
        }
        combined, _breakdown = hybrid_ranker.combine(component_scores)
        return combined

    report = {
        "popularity": evaluate_rankings(
            popularity_scorer,
            split.train,
            split.test,
            bundle.movies,
            top_k=top_k,
            candidate_movie_ids=ranking_movie_ids,
        ),
        "content": evaluate_rankings(
            content_scorer,
            split.train,
            split.test,
            bundle.movies,
            top_k=top_k,
            candidate_movie_ids=ranking_movie_ids,
        ),
        "audience": evaluate_rankings(
            audience_scorer,
            split.train,
            split.test,
            bundle.movies,
            top_k=top_k,
            candidate_movie_ids=ranking_movie_ids,
        ),
        "svd": {
            **evaluate_explicit_model(bundle.svd_model, split.test),
            **evaluate_rankings(
                model_scorer(bundle.svd_model),
                split.train,
                split.test,
                bundle.movies,
                top_k=top_k,
                candidate_movie_ids=ranking_movie_ids,
            ),
        },
        "matrix_factorization": {
            **evaluate_explicit_model(bundle.matrix_factorization_model, split.test),
            **evaluate_rankings(
                model_scorer(bundle.matrix_factorization_model),
                split.train,
                split.test,
                bundle.movies,
                top_k=top_k,
                candidate_movie_ids=ranking_movie_ids,
            ),
        },
        "autoencoder": {
            **evaluate_explicit_model(bundle.autoencoder_model, split.test),
            **evaluate_rankings(
                model_scorer(bundle.autoencoder_model),
                split.train,
                split.test,
                bundle.movies,
                top_k=top_k,
                candidate_movie_ids=ranking_movie_ids,
            ),
        },
        "hybrid": evaluate_rankings(
            hybrid_scorer,
            split.train,
            split.test,
            bundle.movies,
            top_k=top_k,
            candidate_movie_ids=ranking_movie_ids,
        ),
    }
    return report
