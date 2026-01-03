from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from movie_recommender.recommenders.base import UserItemScoreModel


def fit_svd_recommender(
    ratings: pd.DataFrame,
    n_components: int = 40,
    random_state: int = 42,
) -> UserItemScoreModel:
    matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0).sort_index()
    values = matrix.to_numpy(dtype=float)
    mask = (values > 0).astype(float)
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    user_means = values.sum(axis=1, keepdims=True) / counts
    demeaned = np.where(mask > 0, values - user_means, 0.0)

    n_components = max(2, min(n_components, min(demeaned.shape) - 1))
    model = TruncatedSVD(n_components=n_components, random_state=random_state)
    latent = model.fit_transform(demeaned)
    reconstructed = np.dot(latent, model.components_) + user_means

    return UserItemScoreModel(
        name="svd",
        user_ids=matrix.index.astype(int).tolist(),
        movie_ids=matrix.columns.astype(int).tolist(),
        predictions=reconstructed,
        global_mean=float(ratings["rating"].mean()),
    )

