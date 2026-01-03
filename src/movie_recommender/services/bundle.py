from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from movie_recommender.features.content import ContentFeatureStore
from movie_recommender.recommenders.base import UserItemScoreModel
from movie_recommender.recommenders.content import ContentRecommender
from movie_recommender.recommenders.popularity import PopularityRecommender
from movie_recommender.utils.io import load_pickle, save_pickle


@dataclass
class ModelBundle:
    dataset_manifest: dict
    movies: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame
    popularity: PopularityRecommender
    content_features: ContentFeatureStore
    content_recommender: ContentRecommender
    svd_model: UserItemScoreModel
    matrix_factorization_model: UserItemScoreModel
    autoencoder_model: UserItemScoreModel
    hybrid_weights: dict[str, float]
    user_profiles: dict[int, dict]
    evaluation_report: dict = field(default_factory=dict)

    def save(self, path) -> None:
        save_pickle(self, path)

    @classmethod
    def load(cls, path) -> "ModelBundle":
        return load_pickle(path)

