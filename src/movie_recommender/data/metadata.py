from __future__ import annotations

from pathlib import Path

import pandas as pd

from movie_recommender.utils.io import load_json


def load_optional_metadata(path: Path) -> pd.DataFrame:
    payload = load_json(path, default={})
    if not payload:
        return pd.DataFrame(columns=["movie_id", "overview", "poster_url"])

    records: list[dict] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            record = dict(value or {})
            record["movie_id"] = int(record.get("movie_id") or key)
            records.append(record)
    else:
        for item in payload:
            record = dict(item)
            if "movie_id" not in record and "movieId" in record:
                record["movie_id"] = int(record["movieId"])
            records.append(record)

    metadata = pd.DataFrame.from_records(records)
    if metadata.empty:
        return pd.DataFrame(columns=["movie_id", "overview", "poster_url"])

    metadata = metadata.rename(columns={"movieId": "movie_id", "posterUrl": "poster_url"})
    for column in ["overview", "poster_url"]:
        if column not in metadata:
            metadata[column] = ""
        metadata[column] = metadata[column].fillna("").astype(str)
    if "tmdb_id" not in metadata.columns:
        metadata["tmdb_id"] = pd.Series(dtype="float64")
    return metadata[["movie_id", "tmdb_id", "overview", "poster_url"]].drop_duplicates(subset=["movie_id"])

