from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from movie_recommender.config.settings import Settings
from movie_recommender.data.catalog import prepare_catalog_movies
from movie_recommender.data.download import download_movielens_dataset
from movie_recommender.data.metadata import load_optional_metadata
from movie_recommender.data.schemas import PreparedDataBundle
from movie_recommender.ranking.audience import enrich_movie_audience_signals
from movie_recommender.utils.io import ensure_dir, load_json, save_json

LOGGER = logging.getLogger(__name__)

TITLE_YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def _time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _parse_movie_title(raw_title: str) -> tuple[str, float]:
    match = TITLE_YEAR_PATTERN.search(raw_title or "")
    year = float(match.group(1)) if match else np.nan
    clean_title = TITLE_YEAR_PATTERN.sub("", raw_title or "").strip()
    return clean_title or raw_title or "Unknown", year


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _normalize_links(links: pd.DataFrame) -> pd.DataFrame:
    links = links.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}).copy()
    if links.empty:
        return pd.DataFrame(columns=["movie_id", "imdb_tconst", "tmdb_id"])
    links["movie_id"] = pd.to_numeric(links["movie_id"], errors="coerce").astype("Int64")
    links["tmdb_id"] = pd.to_numeric(links.get("tmdb_id"), errors="coerce")
    imdb_numeric = pd.to_numeric(links.get("imdb_id"), errors="coerce")
    links["imdb_tconst"] = imdb_numeric.map(lambda value: f"tt{int(value):07d}" if pd.notna(value) else None)
    return links[["movie_id", "imdb_tconst", "tmdb_id"]].drop_duplicates(subset=["movie_id"])


def _finalize_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    genre_dummies = movies["genres"].str.get_dummies(sep="|").add_prefix("genre_")
    movies = pd.concat([movies.drop(columns=[column for column in movies.columns if column.startswith("genre_")], errors="ignore"), genre_dummies], axis=1)
    scaler = MinMaxScaler()
    numeric_features = movies[["avg_rating", "rating_count", "year"]].astype(float)
    scaled = scaler.fit_transform(numeric_features)
    movies["avg_rating_scaled"] = scaled[:, 0]
    movies["rating_count_scaled"] = scaled[:, 1]
    movies["year_scaled"] = scaled[:, 2]
    return movies


def _normalize_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    movies = movies.rename(columns={"movieId": "movie_id"}).copy()
    movies["title"] = movies["title"].fillna("Unknown").astype(str)
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)
    parsed = movies["title"].apply(_parse_movie_title)
    movies["clean_title"] = parsed.map(lambda item: item[0])
    movies["year"] = parsed.map(lambda item: item[1])
    movies["year"] = movies["year"].fillna(movies["year"].median())

    movie_tag_text = (
        tags.groupby("movie_id")["tag"]
        .agg(lambda values: " ".join(sorted({str(value).strip().lower() for value in values if str(value).strip()})))
        .rename("movie_tag_text")
    )
    movie_stats = (
        ratings.groupby("movie_id")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )
    movies = movies.merge(movie_stats, on="movie_id", how="left").merge(movie_tag_text, on="movie_id", how="left")
    if links is not None and not links.empty:
        movies = movies.merge(links, on="movie_id", how="left")
    else:
        movies["imdb_tconst"] = pd.Series(dtype="object")
        movies["tmdb_id"] = pd.Series(dtype="float64")
    movies["avg_rating"] = movies["avg_rating"].fillna(ratings["rating"].mean())
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    movies["movie_tag_text"] = movies["movie_tag_text"].fillna("")
    movies["source_catalog"] = "interactions"
    return enrich_movie_audience_signals(movies, ratings=ratings)


def _normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"}).copy()
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id", "timestamp"])
    ratings["timestamp"] = ratings["timestamp"].fillna(0).astype(int)
    ratings["rating"] = ratings["rating"].fillna(ratings["rating"].median()).astype(float)
    ratings["watched_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["hour"] = ratings["watched_at"].dt.hour
    ratings["day_of_week"] = ratings["watched_at"].dt.day_name()
    ratings["time_of_day"] = ratings["hour"].map(_time_of_day)
    rating_mean = ratings["rating"].mean()
    rating_std = ratings["rating"].std() or 1.0
    ratings["rating_normalized"] = (ratings["rating"] - rating_mean) / rating_std
    return ratings


def _normalize_tags(tags: pd.DataFrame) -> pd.DataFrame:
    tags = tags.rename(columns={"userId": "user_id", "movieId": "movie_id"}).copy()
    if tags.empty:
        tags = pd.DataFrame(columns=["user_id", "movie_id", "tag", "timestamp", "tagged_at"])
        return tags
    tags["tag"] = tags["tag"].fillna("").astype(str).str.strip().str.lower()
    tags = tags[tags["tag"] != ""].drop_duplicates(subset=["user_id", "movie_id", "tag"])
    tags["timestamp"] = tags["timestamp"].fillna(0).astype(int)
    tags["tagged_at"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)
    return tags


def _build_users(ratings: pd.DataFrame, tags: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    genre_columns = [column for column in movies.columns if column.startswith("genre_")]
    ratings_with_movies = ratings.merge(movies[["movie_id", *genre_columns]], on="movie_id", how="left")
    user_stats = ratings.groupby("user_id")["rating"].agg(avg_rating_given="mean", rating_count="count").reset_index()
    user_tag_text = (
        tags.groupby("user_id")["tag"]
        .agg(lambda values: " ".join(sorted({str(value) for value in values if str(value).strip()})))
        .rename("user_tag_text")
        .reset_index()
    )
    genre_preferences = (
        ratings_with_movies.groupby("user_id")[genre_columns]
        .mean()
        .fillna(0.0)
        .reset_index()
    )
    users = user_stats.merge(user_tag_text, on="user_id", how="left").merge(genre_preferences, on="user_id", how="left")
    users["user_tag_text"] = users["user_tag_text"].fillna("")
    users["activity_scaled"] = MinMaxScaler().fit_transform(users[["rating_count"]].astype(float)).ravel()
    users["avg_rating_given"] = users["avg_rating_given"].fillna(ratings["rating"].mean())
    return users


def _generate_llm_summaries(
    movies: pd.DataFrame,
    llm_backend: object | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    if llm_backend is None:
        movies["llm_summary"] = ""
        return movies

    summaries: list[str] = []
    for index, row in movies.iterrows():
        if index >= limit:
            summaries.append("")
            continue
        summary = ""
        try:
            summary = llm_backend.summarize_movie(
                title=row["clean_title"],
                genres=row["genres"],
                overview=row.get("overview", ""),
                tags=row.get("movie_tag_text", ""),
            )
        except Exception:
            summary = ""
        summaries.append(summary)
    if len(summaries) < len(movies):
        summaries.extend([""] * (len(movies) - len(summaries)))
    movies["llm_summary"] = summaries
    return movies


def prepare_dataset(
    settings: Settings,
    force_download: bool = False,
    llm_backend: object | None = None,
    generate_summaries: bool = False,
) -> PreparedDataBundle:
    source_dir = download_movielens_dataset(settings, force=force_download)
    processed_dir = ensure_dir(settings.processed_data_root / settings.dataset_name / settings.processed_version)

    ratings = _normalize_ratings(_load_csv(source_dir / "ratings.csv"))
    tags_path = source_dir / "tags.csv"
    tags = _normalize_tags(_load_csv(tags_path)) if tags_path.exists() else pd.DataFrame(columns=["user_id", "movie_id", "tag", "timestamp"])
    links_path = source_dir / "links.csv"
    links = _normalize_links(_load_csv(links_path)) if links_path.exists() else pd.DataFrame(columns=["movie_id", "imdb_tconst", "tmdb_id"])
    movies = _normalize_movies(_load_csv(source_dir / "movies.csv"), ratings, tags, links=links)

    catalog_movies = prepare_catalog_movies(settings, movies)
    if not catalog_movies.empty:
        for column in ["imdb_tconst", "tmdb_id", "source_catalog"]:
            if column not in movies.columns:
                movies[column] = pd.Series(dtype="object")
        movies = pd.concat([movies, catalog_movies], ignore_index=True, sort=False)

    metadata = load_optional_metadata(settings.metadata_path)
    if not metadata.empty:
        metadata = metadata.rename(
            columns={
                "tmdb_id": "metadata_tmdb_id",
                "overview": "metadata_overview",
                "poster_url": "metadata_poster_url",
            }
        )
        movies = movies.merge(metadata, on="movie_id", how="left")
    if "tmdb_id" not in movies:
        movies["tmdb_id"] = pd.Series(dtype="float64")
    if "metadata_tmdb_id" in movies:
        movies["tmdb_id"] = movies["tmdb_id"].combine_first(movies["metadata_tmdb_id"])
    if "overview" not in movies:
        movies["overview"] = ""
    if "metadata_overview" in movies:
        movies["overview"] = movies["overview"].where(movies["overview"].astype(str).str.strip() != "", np.nan).combine_first(movies["metadata_overview"])
    if "poster_url" not in movies:
        movies["poster_url"] = ""
    if "metadata_poster_url" in movies:
        movies["poster_url"] = movies["poster_url"].where(movies["poster_url"].astype(str).str.strip() != "", np.nan).combine_first(movies["metadata_poster_url"])
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["poster_url"] = movies["poster_url"].fillna("").astype(str)
    movies = movies.drop(columns=["metadata_tmdb_id", "metadata_overview", "metadata_poster_url"], errors="ignore")
    if "imdb_tconst" not in movies:
        movies["imdb_tconst"] = pd.Series(dtype="object")
    if "source_catalog" not in movies:
        movies["source_catalog"] = "interactions"

    if generate_summaries:
        movies = _generate_llm_summaries(movies, llm_backend=llm_backend)
    else:
        movies["llm_summary"] = ""

    movies = enrich_movie_audience_signals(movies, ratings=ratings)
    movies["combined_text"] = (
        movies["clean_title"].fillna("")
        + " "
        + movies["genres"].str.replace("|", " ", regex=False).fillna("")
        + " "
        + movies["movie_tag_text"].fillna("")
        + " "
        + movies["audience_review_text"].fillna("")
        + " "
        + movies["overview"].fillna("")
        + " "
        + movies["llm_summary"].fillna("")
    ).str.strip()
    movies = _finalize_movie_features(movies)

    users = _build_users(ratings, tags, movies)

    ratings.to_csv(processed_dir / "ratings.csv", index=False)
    movies.to_csv(processed_dir / "movies.csv", index=False)
    tags.to_csv(processed_dir / "tags.csv", index=False)
    users.to_csv(processed_dir / "users.csv", index=False)

    manifest = {
        "dataset_name": settings.dataset_name,
        "dataset_url": settings.dataset_url,
        "processed_version": settings.processed_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": {
            "ratings": int(len(ratings)),
            "movies": int(len(movies)),
            "tags": int(len(tags)),
            "users": int(len(users)),
        },
        "columns": {
            "ratings": ratings.columns.tolist(),
            "movies": movies.columns.tolist(),
            "tags": tags.columns.tolist(),
            "users": users.columns.tolist(),
        },
        "llm_summaries": bool(generate_summaries),
        "metadata_enriched": bool(metadata.shape[0]),
        "catalog_source": settings.catalog_source,
        "catalog_limit": settings.catalog_limit,
        "catalog_movies": int((movies.get("source_catalog", "interactions") == "imdb").sum()) if "source_catalog" in movies else 0,
    }
    save_json(manifest, processed_dir / "manifest.json")
    LOGGER.info("Prepared dataset", extra={"context": manifest["counts"]})

    return PreparedDataBundle(
        data_dir=processed_dir,
        ratings=ratings,
        movies=movies,
        tags=tags,
        users=users,
        manifest=manifest,
    )


def load_processed_bundle(settings: Settings) -> PreparedDataBundle:
    processed_dir = settings.processed_data_root / settings.dataset_name / settings.processed_version
    manifest_path = processed_dir / "manifest.json"
    ratings = pd.read_csv(processed_dir / "ratings.csv", parse_dates=["watched_at"])
    movies = pd.read_csv(processed_dir / "movies.csv")
    tags_path = processed_dir / "tags.csv"
    tags = pd.read_csv(tags_path, parse_dates=["tagged_at"]) if tags_path.exists() else pd.DataFrame()
    users = pd.read_csv(processed_dir / "users.csv")
    manifest = load_json(manifest_path, default={})
    return PreparedDataBundle(processed_dir, ratings, movies, tags, users, manifest)
