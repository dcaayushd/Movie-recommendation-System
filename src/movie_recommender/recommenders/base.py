from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UserItemScoreModel:
    name: str
    user_ids: list[int]
    movie_ids: list[int]
    predictions: np.ndarray
    global_mean: float
    user_id_to_index: dict[int, int] = field(init=False)
    movie_id_to_index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.user_id_to_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.movie_id_to_index = {movie_id: index for index, movie_id in enumerate(self.movie_ids)}

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_id_to_index or movie_id not in self.movie_id_to_index:
            return float(self.global_mean)
        return float(self.predictions[self.user_id_to_index[user_id], self.movie_id_to_index[movie_id]])

    def score_user(self, user_id: int, candidate_movie_ids: list[int]) -> dict[int, float]:
        return {movie_id: self.predict(user_id, movie_id) for movie_id in candidate_movie_ids}

