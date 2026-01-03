from __future__ import annotations

import pytest

from movie_recommender.models.autoencoder import AutoEncoderConfig
from movie_recommender.models.matrix_factorization import MatrixFactorizationConfig
from movie_recommender.services.inference import MovieRecommenderService
from movie_recommender.services.training import TrainingConfig, train_model_bundle
from tests.helpers import build_prepared_bundle, build_settings


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("movie_recommender")
    return build_settings(root)


@pytest.fixture(scope="session")
def prepared_bundle(settings: Settings) -> PreparedDataBundle:
    return build_prepared_bundle(settings)


@pytest.fixture(scope="session")
def trained_bundle(settings: Settings, prepared_bundle: PreparedDataBundle):
    config = TrainingConfig(
        use_semantic_embeddings=False,
        top_k=5,
        mf_config=MatrixFactorizationConfig(embedding_dim=8, epochs=2, batch_size=4),
        autoencoder_config=AutoEncoderConfig(hidden_dim=8, epochs=2, batch_size=2),
    )
    bundle, _split = train_model_bundle(settings, prepared_bundle, training_config=config)
    return bundle


@pytest.fixture
def service(settings: Settings, trained_bundle):
    return MovieRecommenderService.from_path(settings)
