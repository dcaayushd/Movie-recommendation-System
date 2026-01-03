from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(max_value, min_value):
        return {key: 0.0 for key in scores}
    return {key: float((value - min_value) / (max_value - min_value)) for key, value in scores.items()}


@dataclass
class HybridRanker:
    weights: dict[str, float]

    def combine(self, score_maps: dict[str, dict[int, float]]) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
        normalized = {name: min_max_normalize(scores) for name, scores in score_maps.items()}
        all_movie_ids = sorted({movie_id for scores in normalized.values() for movie_id in scores})
        combined: dict[int, float] = {}
        breakdown: dict[int, dict[str, float]] = {}
        for movie_id in all_movie_ids:
            partials = {}
            total = 0.0
            for name, scores in normalized.items():
                weighted_score = float(self.weights.get(name, 0.0)) * float(scores.get(movie_id, 0.0))
                partials[name] = weighted_score
                total += weighted_score
            combined[movie_id] = total
            breakdown[movie_id] = partials
        return combined, breakdown


def weight_candidates() -> list[dict[str, float]]:
    return [
        {"popularity": 0.08, "content": 0.28, "audience": 0.22, "svd": 0.17, "matrix_factorization": 0.17, "autoencoder": 0.08},
        {"popularity": 0.06, "content": 0.34, "audience": 0.24, "svd": 0.14, "matrix_factorization": 0.14, "autoencoder": 0.08},
        {"popularity": 0.10, "content": 0.24, "audience": 0.26, "svd": 0.15, "matrix_factorization": 0.17, "autoencoder": 0.08},
        {"popularity": 0.06, "content": 0.25, "audience": 0.19, "svd": 0.20, "matrix_factorization": 0.20, "autoencoder": 0.10},
        {"popularity": 0.05, "content": 0.22, "audience": 0.23, "svd": 0.18, "matrix_factorization": 0.22, "autoencoder": 0.10},
        {"popularity": 0.04, "content": 0.20, "audience": 0.21, "svd": 0.20, "matrix_factorization": 0.20, "autoencoder": 0.15},
    ]
