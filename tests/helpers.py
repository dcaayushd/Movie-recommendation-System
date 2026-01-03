from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from movie_recommender.config.settings import Settings
from movie_recommender.data.metadata import load_optional_metadata
from movie_recommender.data.preprocess import _normalize_movies, _normalize_ratings, _normalize_tags
from movie_recommender.data.schemas import PreparedDataBundle
from movie_recommender.ranking.audience import enrich_movie_audience_signals


def build_settings(root: Path) -> Settings:
    data_root = root / "data"
    raw_data_root = data_root / "raw"
    processed_data_root = data_root / "processed"
    metadata_root = data_root / "metadata"
    artifacts_root = root / "artifacts"
    mlruns_root = root / "mlruns"

    for path in [raw_data_root, processed_data_root, metadata_root, artifacts_root, mlruns_root]:
        path.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_root / "sample_movie_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "1": {"overview": "Animated toys come alive.", "poster_url": "https://example.com/toy-story.jpg"},
                "2": {"overview": "Jumanji becomes a dangerous game.", "poster_url": "https://example.com/jumanji.jpg"},
            }
        ),
        encoding="utf-8",
    )

    return Settings(
        project_root=root,
        data_root=data_root,
        raw_data_root=raw_data_root,
        processed_data_root=processed_data_root,
        metadata_root=metadata_root,
        artifacts_root=artifacts_root,
        mlruns_root=mlruns_root,
        dataset_name="ml-latest-small",
        dataset_url="https://example.com/ml-latest-small.zip",
        catalog_source="none",
        catalog_limit=None,
        catalog_min_votes=0,
        catalog_include_adult=False,
        processed_version="v1",
        metadata_path=metadata_path,
        bundle_path=artifacts_root / "model_bundle.pkl",
        eval_report_path=artifacts_root / "evaluation_report.json",
        feedback_log_path=artifacts_root / "feedback.jsonl",
        download_ca_bundle=None,
        allow_insecure_download=False,
        default_sentence_model="sentence-transformers/all-MiniLM-L6-v2",
        default_transformer_model="google/flan-t5-small",
        default_ollama_model="llama3",
    )


def build_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children|Comedy"},
            {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
            {"movieId": 3, "title": "Heat (1995)", "genres": "Action|Crime|Thriller"},
            {"movieId": 4, "title": "Sabrina (1995)", "genres": "Comedy|Romance"},
            {"movieId": 5, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller"},
            {"movieId": 6, "title": "Sense and Sensibility (1995)", "genres": "Drama|Romance"},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.5, "timestamp": 964982703},
            {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 964982224},
            {"userId": 1, "movieId": 3, "rating": 2.0, "timestamp": 964983815},
            {"userId": 1, "movieId": 4, "rating": 3.5, "timestamp": 964982931},
            {"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": 964982400},
            {"userId": 2, "movieId": 3, "rating": 4.5, "timestamp": 964982653},
            {"userId": 2, "movieId": 5, "rating": 4.0, "timestamp": 964983200},
            {"userId": 2, "movieId": 6, "rating": 2.5, "timestamp": 964984100},
            {"userId": 3, "movieId": 2, "rating": 4.5, "timestamp": 964982211},
            {"userId": 3, "movieId": 4, "rating": 2.5, "timestamp": 964982545},
            {"userId": 3, "movieId": 5, "rating": 4.0, "timestamp": 964983123},
            {"userId": 3, "movieId": 6, "rating": 4.5, "timestamp": 964984200},
            {"userId": 4, "movieId": 1, "rating": 3.0, "timestamp": 964982111},
            {"userId": 4, "movieId": 3, "rating": 4.5, "timestamp": 964983555},
            {"userId": 4, "movieId": 5, "rating": 3.5, "timestamp": 964983777},
            {"userId": 4, "movieId": 6, "rating": 4.0, "timestamp": 964984333},
        ]
    )
    tags = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "tag": "pixar", "timestamp": 964982703},
            {"userId": 1, "movieId": 2, "tag": "adventure", "timestamp": 964982224},
            {"userId": 2, "movieId": 3, "tag": "crime", "timestamp": 964982653},
            {"userId": 2, "movieId": 5, "tag": "spy", "timestamp": 964983200},
            {"userId": 3, "movieId": 6, "tag": "period drama", "timestamp": 964984200},
        ]
    )
    return movies, ratings, tags


def build_prepared_bundle(settings: Settings) -> PreparedDataBundle:
    raw_movies, raw_ratings, raw_tags = build_raw_frames()
    ratings = _normalize_ratings(raw_ratings)
    tags = _normalize_tags(raw_tags)
    movies = _normalize_movies(raw_movies, ratings, tags)
    metadata = load_optional_metadata(settings.metadata_path)
    metadata = metadata.rename(
        columns={
            "tmdb_id": "metadata_tmdb_id",
            "overview": "metadata_overview",
            "poster_url": "metadata_poster_url",
        }
    )
    movies = movies.merge(metadata, on="movie_id", how="left")
    overview_series = movies.get("overview", pd.Series("", index=movies.index)).fillna("").astype(str)
    poster_series = movies.get("poster_url", pd.Series("", index=movies.index)).fillna("").astype(str)
    metadata_overview = movies.get("metadata_overview", pd.Series("", index=movies.index)).fillna("").astype(str)
    metadata_poster = movies.get("metadata_poster_url", pd.Series("", index=movies.index)).fillna("").astype(str)
    movies["overview"] = overview_series.where(overview_series.str.strip() != "", pd.NA).combine_first(metadata_overview)
    movies["poster_url"] = poster_series.where(poster_series.str.strip() != "", pd.NA).combine_first(metadata_poster)
    if "tmdb_id" in movies.columns and "metadata_tmdb_id" in movies.columns:
        movies["tmdb_id"] = movies["tmdb_id"].combine_first(movies["metadata_tmdb_id"])
    movies = movies.drop(columns=["metadata_tmdb_id", "metadata_overview", "metadata_poster_url"], errors="ignore")
    movies["overview"] = movies["overview"].fillna("")
    movies["poster_url"] = movies["poster_url"].fillna("")
    movies["llm_summary"] = ""
    movies = enrich_movie_audience_signals(movies, ratings=ratings)
    movies["combined_text"] = (
        movies["clean_title"]
        + " "
        + movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["movie_tag_text"].fillna("")
        + " "
        + movies["audience_review_text"].fillna("")
        + " "
        + movies["overview"].fillna("")
    ).str.strip()
    users = pd.DataFrame({"user_id": sorted(ratings["user_id"].unique())})
    return PreparedDataBundle(
        data_dir=settings.processed_data_root / settings.dataset_name / settings.processed_version,
        ratings=ratings,
        movies=movies,
        tags=tags,
        users=users,
        manifest={"dataset_name": settings.dataset_name, "counts": {"ratings": len(ratings), "movies": len(movies)}},
    )
