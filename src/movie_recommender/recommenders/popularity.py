from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PopularityRecommender:
    movie_scores: dict[int, float]
    global_mean: float
    vote_floor: float

    @classmethod
    def fit(cls, ratings: pd.DataFrame) -> "PopularityRecommender":
        grouped = ratings.groupby("movie_id")["rating"].agg(avg_rating="mean", rating_count="count").reset_index()
        global_mean = float(ratings["rating"].mean())
        vote_floor = float(grouped["rating_count"].quantile(0.6)) if not grouped.empty else 1.0

        scores: dict[int, float] = {}
        for row in grouped.itertuples(index=False):
            weighted = (row.rating_count / (row.rating_count + vote_floor)) * row.avg_rating
            weighted += (vote_floor / (row.rating_count + vote_floor)) * global_mean
            scores[int(row.movie_id)] = float(weighted)

        return cls(movie_scores=scores, global_mean=global_mean, vote_floor=vote_floor)

    def score_movies(self, candidate_movie_ids: list[int]) -> dict[int, float]:
        return {movie_id: float(self.movie_scores.get(movie_id, self.global_mean)) for movie_id in candidate_movie_ids}

