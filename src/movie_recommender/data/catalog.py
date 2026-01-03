from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.config.settings import Settings
from movie_recommender.data.download import _download_with_curl, _download_with_requests, _verify_option
from movie_recommender.ranking.audience import enrich_movie_audience_signals
from movie_recommender.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_TITLE_TYPES = {"movie", "tvMovie", "short", "video"}
IMDB_MOVIE_ID_OFFSET = 100_000_000


def _catalog_root(settings: Settings) -> Path:
    return ensure_dir(settings.raw_data_root / "catalog")


def _download_file(url: str, target_path: Path, settings: Settings) -> None:
    verify = _verify_option(settings)
    try:
        _download_with_requests(url, target_path, verify=verify)
        return
    except Exception as exc:
        LOGGER.warning(
            "Primary catalog download failed; retrying with curl",
            extra={"context": {"url": url, "error": str(exc)}},
        )
    _download_with_curl(url, target_path, settings)


def download_imdb_catalog(settings: Settings, force: bool = False) -> dict[str, Path]:
    catalog_root = _catalog_root(settings)
    basics_path = catalog_root / "title.basics.tsv.gz"
    ratings_path = catalog_root / "title.ratings.tsv.gz"

    if force:
        basics_path.unlink(missing_ok=True)
        ratings_path.unlink(missing_ok=True)

    if not basics_path.exists():
        LOGGER.info("Downloading global movie catalog basics", extra={"context": {"url": IMDB_BASICS_URL}})
        _download_file(IMDB_BASICS_URL, basics_path, settings)
    if not ratings_path.exists():
        LOGGER.info("Downloading global movie catalog ratings", extra={"context": {"url": IMDB_RATINGS_URL}})
        _download_file(IMDB_RATINGS_URL, ratings_path, settings)

    return {"basics": basics_path, "ratings": ratings_path}


def _title_key(frame: pd.DataFrame) -> pd.Series:
    clean_title = frame["clean_title"].fillna("").astype(str).str.strip().str.lower() if "clean_title" in frame else pd.Series("", index=frame.index)
    year = frame["year"].fillna(-1).astype(float).round().astype(int).astype(str) if "year" in frame else pd.Series("-1", index=frame.index)
    return clean_title + "::" + year


def _imdb_movie_id(tconst: str) -> int:
    digits = "".join(character for character in str(tconst) if character.isdigit())
    return IMDB_MOVIE_ID_OFFSET + int(digits or 0)


def _normalize_imdb_rating(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value) / 2.0


def prepare_catalog_movies(settings: Settings, existing_movies: pd.DataFrame) -> pd.DataFrame:
    if settings.catalog_source == "none":
        return pd.DataFrame()
    if settings.catalog_source != "imdb":
        raise ValueError(f"Unsupported catalog source '{settings.catalog_source}'.")

    paths = download_imdb_catalog(settings)
    basics = pd.read_csv(
        paths["basics"],
        sep="\t",
        compression="gzip",
        na_values="\\N",
        dtype={"tconst": str, "titleType": str, "primaryTitle": str, "isAdult": str, "genres": str},
        usecols=["tconst", "titleType", "primaryTitle", "isAdult", "startYear", "genres"],
    )
    ratings = pd.read_csv(
        paths["ratings"],
        sep="\t",
        compression="gzip",
        na_values="\\N",
        dtype={"tconst": str},
        usecols=["tconst", "averageRating", "numVotes"],
    )

    catalog = basics.merge(ratings, on="tconst", how="left")
    catalog = catalog[catalog["titleType"].isin(IMDB_TITLE_TYPES)].copy()
    if not settings.catalog_include_adult:
        catalog = catalog[catalog["isAdult"].fillna("0").astype(str) != "1"]
    catalog["primaryTitle"] = catalog["primaryTitle"].fillna("Unknown").astype(str)
    catalog["startYear"] = pd.to_numeric(catalog["startYear"], errors="coerce")
    catalog["averageRating"] = pd.to_numeric(catalog["averageRating"], errors="coerce")
    catalog["numVotes"] = pd.to_numeric(catalog["numVotes"], errors="coerce").fillna(0).astype(int)
    catalog["genres"] = catalog["genres"].fillna("(no genres listed)").astype(str).str.replace(",", "|", regex=False)
    catalog = catalog[catalog["numVotes"] >= settings.catalog_min_votes].copy()
    catalog = catalog.sort_values(["numVotes", "averageRating"], ascending=[False, False])
    if settings.catalog_limit is not None:
        catalog = catalog.head(settings.catalog_limit).copy()

    existing_imdb_ids = set(existing_movies.get("imdb_tconst", pd.Series(dtype=str)).dropna().astype(str))
    if existing_imdb_ids:
        catalog = catalog[~catalog["tconst"].isin(existing_imdb_ids)].copy()

    existing_keys = set(_title_key(existing_movies))
    catalog["clean_title"] = catalog["primaryTitle"].astype(str).str.strip()
    catalog["year"] = catalog["startYear"].fillna(np.nan)
    catalog["catalog_key"] = _title_key(catalog)
    if existing_keys:
        catalog = catalog[~catalog["catalog_key"].isin(existing_keys)].copy()
    catalog = catalog.drop_duplicates(subset=["tconst"])

    if catalog.empty:
        return pd.DataFrame()

    title_year = catalog["startYear"].fillna(np.nan)
    title_suffix = pd.Series("", index=catalog.index, dtype="object")
    valid_years = title_year.notna()
    if valid_years.any():
        title_suffix.loc[valid_years] = " (" + title_year.loc[valid_years].round().astype(int).astype(str) + ")"
    title_text = catalog["primaryTitle"].astype(str) + title_suffix
    movies = pd.DataFrame(
        {
            "movie_id": catalog["tconst"].map(_imdb_movie_id).astype(int),
            "title": title_text,
            "clean_title": catalog["primaryTitle"].astype(str),
            "year": title_year,
            "genres": catalog["genres"].astype(str),
            "avg_rating": catalog["averageRating"].map(_normalize_imdb_rating).fillna(
                existing_movies["avg_rating"].mean() if not existing_movies.empty else 0.0
            ),
            "rating_count": catalog["numVotes"].fillna(0).astype(int),
            "movie_tag_text": "",
            "imdb_tconst": catalog["tconst"].astype(str),
            "tmdb_id": np.nan,
            "source_catalog": "imdb",
            "overview": "",
            "poster_url": "",
            "llm_summary": "",
        }
    )
    movies = enrich_movie_audience_signals(movies)
    movies["combined_text"] = (
        movies["clean_title"].fillna("")
        + " "
        + movies["genres"].str.replace("|", " ", regex=False).fillna("")
        + " "
        + movies["audience_review_text"].fillna("")
    ).str.strip()
    LOGGER.info("Prepared external movie catalog", extra={"context": {"movies": int(len(movies)), "source": settings.catalog_source}})
    return movies
