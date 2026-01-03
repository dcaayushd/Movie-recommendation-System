from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from movie_recommender.config.settings import Settings
from movie_recommender.data.schemas import PreparedDataBundle
from movie_recommender.features.content import fit_content_features, profile_embedding
from movie_recommender.models.autoencoder import AutoEncoderConfig, fit_autoencoder
from movie_recommender.models.matrix_factorization import MatrixFactorizationConfig, fit_matrix_factorization
from movie_recommender.ranking.audience import AudienceRecommender
from movie_recommender.recommenders.content import ContentRecommender
from movie_recommender.recommenders.hybrid import HybridRanker, weight_candidates
from movie_recommender.recommenders.popularity import PopularityRecommender
from movie_recommender.recommenders.svd import fit_svd_recommender
from movie_recommender.services.bundle import ModelBundle
from movie_recommender.services.evaluation import RatingsSplit, chronological_split, evaluate_bundle, ndcg_at_k
from movie_recommender.utils.io import save_json

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    use_semantic_embeddings: bool = True
    semantic_movie_limit: int = 50000
    top_k: int = 10
    mf_config: MatrixFactorizationConfig = field(default_factory=MatrixFactorizationConfig)
    autoencoder_config: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)


def _build_user_profiles(
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_recommender: ContentRecommender,
) -> dict[int, dict]:
    genre_columns = [column for column in movies.columns if column.startswith("genre_")]
    ratings_with_movies = train_ratings.merge(movies[["movie_id", "time_of_day"] if "time_of_day" in movies.columns else ["movie_id"]], on="movie_id", how="left")
    movie_frame = movies.set_index("movie_id")
    profiles: dict[int, dict] = {}

    for user_id, frame in train_ratings.groupby("user_id"):
        liked = frame[frame["rating"] >= 4.0]["movie_id"].astype(int).tolist()
        disliked = frame[frame["rating"] <= 2.5]["movie_id"].astype(int).tolist()
        profile_vector = profile_embedding(content_recommender.store, liked or frame.sort_values("rating", ascending=False).head(5)["movie_id"].tolist())
        profile = {
            "liked_movie_ids": liked,
            "disliked_movie_ids": disliked,
            "profile_vector": profile_vector,
            "avg_rating": float(frame["rating"].mean()),
            "time_preferences": frame["time_of_day"].value_counts(normalize=True).to_dict(),
        }
        genre_preferences: dict[str, float] = {}
        if liked:
            liked_genres = movie_frame.loc[[movie_id for movie_id in liked if movie_id in movie_frame.index], genre_columns]
            if not liked_genres.empty:
                genre_preferences = liked_genres.mean().to_dict()
        profile["genre_preferences"] = genre_preferences
        profiles[int(user_id)] = profile
    return profiles


def _validation_ndcg(
    weight_map: dict[str, float],
    bundle: ModelBundle,
    audience_recommender: AudienceRecommender,
    split: RatingsSplit,
    top_k: int,
    ranking_movie_ids: list[int],
) -> float:
    seen_by_user = split.train.groupby("user_id")["movie_id"].agg(set).to_dict()
    valid_user_ids = split.valid["user_id"].astype(int).unique().tolist()
    candidate_pool_by_user = {
        user_id: [movie_id for movie_id in ranking_movie_ids if movie_id not in seen_by_user.get(user_id, set())]
        for user_id in valid_user_ids
    }
    ranker = HybridRanker(weight_map)
    scores: list[float] = []

    for row in split.valid.itertuples(index=False):
        user_id = int(row.user_id)
        relevant = {int(row.movie_id)}
        candidates = candidate_pool_by_user.get(user_id, ranking_movie_ids)
        if int(row.movie_id) not in candidates:
            candidates = candidates + [int(row.movie_id)]
        content_profile = bundle.user_profiles.get(user_id, {})
        component_scores = {
            "popularity": bundle.popularity.score_movies(candidates),
            "content": bundle.content_recommender.score_candidates(
                liked_movie_ids=content_profile.get("liked_movie_ids", []),
                disliked_movie_ids=content_profile.get("disliked_movie_ids", []),
                candidate_movie_ids=candidates,
            ),
            "audience": audience_recommender.score_candidates(
                liked_movie_ids=content_profile.get("liked_movie_ids", []),
                disliked_movie_ids=content_profile.get("disliked_movie_ids", []),
                candidate_movie_ids=candidates,
            ),
            "svd": bundle.svd_model.score_user(user_id, candidates),
            "matrix_factorization": bundle.matrix_factorization_model.score_user(user_id, candidates),
            "autoencoder": bundle.autoencoder_model.score_user(user_id, candidates),
        }
        combined, _breakdown = ranker.combine(component_scores)
        ranked = [movie_id for movie_id, _score in sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        scores.append(ndcg_at_k(ranked, relevant, top_k))

    return float(np.mean(scores)) if scores else 0.0


def _log_mlflow_metrics(settings: Settings, report: dict, training_config: TrainingConfig, use_semantic_embeddings: bool) -> None:
    try:
        import mlflow
    except Exception as exc:
        LOGGER.warning("MLflow unavailable; skipping tracking", extra={"context": {"error": str(exc)}})
        return

    mlflow.set_tracking_uri(settings.mlruns_root.as_uri())
    mlflow.set_experiment("movie-recommender")
    with mlflow.start_run(run_name="hybrid-training"):
        mlflow.log_params(
            {
                "dataset_name": settings.dataset_name,
                "processed_version": settings.processed_version,
                "semantic_embeddings": use_semantic_embeddings,
                "mf_embedding_dim": training_config.mf_config.embedding_dim,
                "autoencoder_hidden_dim": training_config.autoencoder_config.hidden_dim,
            }
        )
        for model_name, metrics in report.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}".replace("@", "_at_"), metric_value)


def train_model_bundle(
    settings: Settings,
    prepared: PreparedDataBundle,
    training_config: TrainingConfig | None = None,
) -> tuple[ModelBundle, RatingsSplit]:
    training_config = training_config or TrainingConfig()
    split = chronological_split(prepared.ratings)
    use_semantic_embeddings = training_config.use_semantic_embeddings and len(prepared.movies) <= training_config.semantic_movie_limit
    ranking_movie_ids = sorted(
        set(split.train["movie_id"].astype(int))
        | set(split.valid["movie_id"].astype(int))
        | set(split.test["movie_id"].astype(int))
    )

    LOGGER.info(
        "Starting model training",
        extra={
            "context": {
                "movies": int(len(prepared.movies)),
                "ratings": int(len(prepared.ratings)),
                "ranking_movies": len(ranking_movie_ids),
                "semantic_embeddings": use_semantic_embeddings,
            }
        },
    )

    LOGGER.info("Fitting popularity baseline")
    popularity = PopularityRecommender.fit(split.train)
    LOGGER.info(
        "Building content features",
        extra={"context": {"movies": int(len(prepared.movies)), "semantic_embeddings": use_semantic_embeddings}},
    )
    content_features = fit_content_features(
        prepared.movies,
        sentence_model_name=settings.default_sentence_model,
        use_semantic=use_semantic_embeddings,
    )
    content_recommender = ContentRecommender(content_features, prepared.movies)
    LOGGER.info("Fitting collaborative SVD model")
    svd_model = fit_svd_recommender(split.train)
    LOGGER.info("Fitting matrix factorization model")
    mf_model = fit_matrix_factorization(split.train, split.valid, training_config.mf_config)
    LOGGER.info("Fitting autoencoder model")
    autoencoder_model = fit_autoencoder(split.train, split.valid, training_config.autoencoder_config)

    provisional_bundle = ModelBundle(
        dataset_manifest=prepared.manifest,
        movies=prepared.movies,
        ratings=prepared.ratings,
        users=prepared.users,
        popularity=popularity,
        content_features=content_features,
        content_recommender=content_recommender,
        svd_model=svd_model,
        matrix_factorization_model=mf_model,
        autoencoder_model=autoencoder_model,
        hybrid_weights={},
        user_profiles={},
    )
    LOGGER.info("Building user profiles")
    provisional_bundle.user_profiles = _build_user_profiles(split.train, prepared.movies, content_recommender)
    audience_recommender = AudienceRecommender.from_movies(prepared.movies)

    best_weights = weight_candidates()[0]
    best_score = -1.0
    LOGGER.info(
        "Tuning hybrid weights",
        extra={"context": {"candidates": len(weight_candidates()), "ranking_movies": len(ranking_movie_ids)}},
    )
    for candidate_weights in weight_candidates():
        score = _validation_ndcg(candidate_weights, provisional_bundle, audience_recommender, split, training_config.top_k, ranking_movie_ids)
        LOGGER.info("Scored hybrid weights", extra={"context": {"weights": candidate_weights, "ndcg": round(score, 5)}})
        if score > best_score:
            best_score = score
            best_weights = candidate_weights

    bundle = ModelBundle(
        dataset_manifest=prepared.manifest,
        movies=prepared.movies,
        ratings=prepared.ratings,
        users=prepared.users,
        popularity=popularity,
        content_features=content_features,
        content_recommender=content_recommender,
        svd_model=svd_model,
        matrix_factorization_model=mf_model,
        autoencoder_model=autoencoder_model,
        hybrid_weights=best_weights,
        user_profiles=provisional_bundle.user_profiles,
    )
    LOGGER.info("Running final evaluation")
    bundle.evaluation_report = evaluate_bundle(bundle, split, top_k=training_config.top_k)

    bundle.save(settings.bundle_path)
    save_json(bundle.evaluation_report, settings.eval_report_path)
    _log_mlflow_metrics(settings, bundle.evaluation_report, training_config, use_semantic_embeddings=use_semantic_embeddings)
    LOGGER.info("Training complete", extra={"context": {"bundle_path": str(settings.bundle_path), "hybrid_weights": best_weights}})
    return bundle, split
