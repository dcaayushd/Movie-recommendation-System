from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from movie_recommender.recommenders.base import UserItemScoreModel


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, n_items: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_items, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_items),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


@dataclass
class AutoEncoderConfig:
    hidden_dim: int = 128
    dropout: float = 0.25
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42


def fit_autoencoder(
    train_ratings: pd.DataFrame,
    valid_ratings: pd.DataFrame | None = None,
    config: AutoEncoderConfig | None = None,
) -> UserItemScoreModel:
    config = config or AutoEncoderConfig()
    torch.manual_seed(config.seed)

    matrix = train_ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0).sort_index()
    user_ids = matrix.index.astype(int).tolist()
    movie_ids = matrix.columns.astype(int).tolist()
    train_matrix = matrix.to_numpy(dtype=np.float32) / 5.0

    model = DenoisingAutoEncoder(len(movie_ids), config.hidden_dim, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(train_matrix), torch.tensor(train_matrix))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if valid_ratings is not None and not valid_ratings.empty:
        valid_matrix = (
            valid_ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0)
            .reindex(index=user_ids, columns=movie_ids, fill_value=0.0)
            .to_numpy(dtype=np.float32)
            / 5.0
        )
        valid_loader = DataLoader(TensorDataset(torch.tensor(valid_matrix), torch.tensor(valid_matrix)), batch_size=config.batch_size)
    else:
        valid_loader = None

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for _epoch in range(config.epochs):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if valid_loader is None:
            continue
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_inputs, batch_targets in valid_loader:
                outputs = model(batch_inputs)
                losses.append(float(criterion(outputs, batch_targets)))
        current_loss = float(np.mean(losses)) if losses else float("inf")
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        reconstructed = model(torch.tensor(train_matrix)).numpy() * 5.0

    return UserItemScoreModel(
        name="autoencoder",
        user_ids=user_ids,
        movie_ids=movie_ids,
        predictions=reconstructed,
        global_mean=float(train_ratings["rating"].mean()),
    )

