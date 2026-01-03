from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from movie_recommender.recommenders.base import UserItemScoreModel


class ExplicitMF(nn.Module):
    def __init__(self, n_users: int, n_movies: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        dot = (user_vec * movie_vec).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_ids) + self.movie_bias(movie_ids)
        return (dot + bias).squeeze(1)


@dataclass
class MatrixFactorizationConfig:
    embedding_dim: int = 32
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    seed: int = 42


def fit_matrix_factorization(
    train_ratings: pd.DataFrame,
    valid_ratings: pd.DataFrame | None = None,
    config: MatrixFactorizationConfig | None = None,
) -> UserItemScoreModel:
    config = config or MatrixFactorizationConfig()
    torch.manual_seed(config.seed)

    matrix = train_ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0).sort_index()
    user_ids = matrix.index.astype(int).tolist()
    movie_ids = matrix.columns.astype(int).tolist()
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}

    def _tensor_dataset(frame: pd.DataFrame) -> TensorDataset:
        users = torch.tensor(frame["user_id"].map(user_id_to_index).to_numpy(), dtype=torch.long)
        movies = torch.tensor(frame["movie_id"].map(movie_id_to_index).to_numpy(), dtype=torch.long)
        ratings = torch.tensor(frame["rating"].to_numpy(), dtype=torch.float32)
        return TensorDataset(users, movies, ratings)

    train_frame = train_ratings[
        train_ratings["user_id"].isin(user_id_to_index) & train_ratings["movie_id"].isin(movie_id_to_index)
    ].copy()
    valid_frame = valid_ratings[
        valid_ratings["user_id"].isin(user_id_to_index) & valid_ratings["movie_id"].isin(movie_id_to_index)
    ].copy() if valid_ratings is not None else None

    model = ExplicitMF(len(user_ids), len(movie_ids), config.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(_tensor_dataset(train_frame), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(_tensor_dataset(valid_frame), batch_size=config.batch_size) if valid_frame is not None and not valid_frame.empty else None

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for _epoch in range(config.epochs):
        model.train()
        for batch_users, batch_movies, batch_ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_users, batch_movies)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()

        if valid_loader is None:
            continue
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_users, batch_movies, batch_ratings in valid_loader:
                predictions = model(batch_users, batch_movies)
                losses.append(float(criterion(predictions, batch_ratings)))
        current_loss = float(np.mean(losses)) if losses else float("inf")
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        user_tensor = torch.arange(len(user_ids)).unsqueeze(1).repeat(1, len(movie_ids)).reshape(-1)
        movie_tensor = torch.arange(len(movie_ids)).repeat(len(user_ids))
        predictions = model(user_tensor, movie_tensor).reshape(len(user_ids), len(movie_ids)).numpy()

    return UserItemScoreModel(
        name="matrix_factorization",
        user_ids=user_ids,
        movie_ids=movie_ids,
        predictions=predictions,
        global_mean=float(train_ratings["rating"].mean()),
    )

