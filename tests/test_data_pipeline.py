from __future__ import annotations

from dataclasses import replace

import pandas as pd

from movie_recommender.data.catalog import prepare_catalog_movies
from movie_recommender.data.preprocess import prepare_dataset

from tests.helpers import build_raw_frames


def test_prepare_dataset_merges_metadata_and_builds_features(settings, monkeypatch, tmp_path):
    raw_dir = tmp_path / "ml-latest-small"
    raw_dir.mkdir(parents=True, exist_ok=True)
    movies, ratings, tags = build_raw_frames()
    movies.to_csv(raw_dir / "movies.csv", index=False)
    ratings.to_csv(raw_dir / "ratings.csv", index=False)
    tags.to_csv(raw_dir / "tags.csv", index=False)

    monkeypatch.setattr("movie_recommender.data.preprocess.download_movielens_dataset", lambda *_args, **_kwargs: raw_dir)
    bundle = prepare_dataset(settings)

    assert bundle.manifest["counts"]["movies"] == 6
    assert bundle.manifest["metadata_enriched"] is True
    assert "combined_text" in bundle.movies.columns
    assert "audience_review_text" in bundle.movies.columns
    assert bundle.movies.loc[bundle.movies["movie_id"] == 1, "overview"].iloc[0] == "Animated toys come alive."
    assert bundle.users["user_id"].nunique() == 4
    assert bundle.ratings["time_of_day"].isin({"morning", "afternoon", "evening", "night"}).all()


def test_prepare_dataset_can_expand_with_external_catalog(settings, monkeypatch, tmp_path):
    raw_dir = tmp_path / "starter"
    raw_dir.mkdir(parents=True, exist_ok=True)
    movies, ratings, tags = build_raw_frames()
    movies.to_csv(raw_dir / "movies.csv", index=False)
    ratings.to_csv(raw_dir / "ratings.csv", index=False)
    tags.to_csv(raw_dir / "tags.csv", index=False)
    pd.DataFrame(
        [
            {"movieId": 1, "imdbId": 114709, "tmdbId": 862},
            {"movieId": 2, "imdbId": 113497, "tmdbId": 8844},
        ]
    ).to_csv(raw_dir / "links.csv", index=False)

    external_catalog = pd.DataFrame(
        [
            {
                "movie_id": 100000123,
                "title": "Arrival (2016)",
                "clean_title": "Arrival",
                "year": 2016.0,
                "genres": "Drama|Sci-Fi",
                "avg_rating": 4.4,
                "rating_count": 500000,
                "movie_tag_text": "",
                "imdb_tconst": "tt2543164",
                "tmdb_id": 329865.0,
                "source_catalog": "imdb",
                "overview": "",
                "poster_url": "",
                "llm_summary": "",
                "combined_text": "Arrival Drama Sci-Fi",
            }
        ]
    )

    monkeypatch.setattr("movie_recommender.data.preprocess.download_movielens_dataset", lambda *_args, **_kwargs: raw_dir)
    monkeypatch.setattr("movie_recommender.data.preprocess.prepare_catalog_movies", lambda *_args, **_kwargs: external_catalog)
    bundle = prepare_dataset(settings)

    assert bundle.manifest["catalog_movies"] == 1
    assert 100000123 in bundle.movies["movie_id"].tolist()
    assert "imdb_tconst" in bundle.movies.columns


def test_prepare_catalog_movies_handles_missing_imdb_year(settings, monkeypatch, tmp_path):
    basics_path = tmp_path / "title.basics.tsv.gz"
    ratings_path = tmp_path / "title.ratings.tsv.gz"

    pd.DataFrame(
        [
            {
                "tconst": "tt1000001",
                "titleType": "movie",
                "primaryTitle": "Unknown Year Movie",
                "isAdult": "0",
                "startYear": None,
                "genres": "Drama",
            },
            {
                "tconst": "tt1000002",
                "titleType": "movie",
                "primaryTitle": "Known Year Movie",
                "isAdult": "0",
                "startYear": 1999,
                "genres": "Comedy",
            },
        ]
    ).to_csv(basics_path, sep="\t", index=False, compression="gzip")
    pd.DataFrame(
        [
            {"tconst": "tt1000001", "averageRating": 7.2, "numVotes": 120},
            {"tconst": "tt1000002", "averageRating": 8.1, "numVotes": 140},
        ]
    ).to_csv(ratings_path, sep="\t", index=False, compression="gzip")

    monkeypatch.setattr(
        "movie_recommender.data.catalog.download_imdb_catalog",
        lambda *_args, **_kwargs: {"basics": basics_path, "ratings": ratings_path},
    )

    catalog_settings = replace(settings, catalog_source="imdb", catalog_min_votes=0, catalog_limit=None)
    existing_movies = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "clean_title": "Toy Story",
                "year": 1995.0,
                "avg_rating": 4.0,
                "imdb_tconst": "tt0114709",
            }
        ]
    )

    movies = prepare_catalog_movies(catalog_settings, existing_movies)

    unknown_year = movies.loc[movies["imdb_tconst"] == "tt1000001"].iloc[0]
    known_year = movies.loc[movies["imdb_tconst"] == "tt1000002"].iloc[0]

    assert unknown_year["title"] == "Unknown Year Movie"
    assert pd.isna(unknown_year["year"])
    assert known_year["title"] == "Known Year Movie (1999)"
    assert round(float(unknown_year["avg_rating"]), 2) == 3.60
    assert round(float(known_year["avg_rating"]), 2) == 4.05
