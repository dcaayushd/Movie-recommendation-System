from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


DATASET_PRESETS = {
    "starter": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "expanded": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "benchmark": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "classic": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}
USER_FACING_PRESETS = ["starter", "expanded", "benchmark", "classic"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def available_dataset_presets() -> list[str]:
    return USER_FACING_PRESETS.copy()


def resolve_dataset(dataset: str | None = None) -> tuple[str, str]:
    dataset = dataset or os.getenv("MOVIE_RECOMMENDER_DATASET", "starter")
    if dataset in DATASET_PRESETS:
        return dataset, DATASET_PRESETS[dataset]
    if dataset.startswith("file://"):
        parsed = urlparse(dataset)
        dataset_path = Path(unquote(parsed.path))
        return dataset_path.stem or "local-dataset", dataset
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists() and dataset_path.is_file():
        return dataset_path.stem or "local-dataset", dataset_path.resolve().as_uri()
    if dataset.startswith(("http://", "https://")):
        parsed = urlparse(dataset)
        filename = Path(parsed.path).name or "custom-dataset.zip"
        dataset_name = Path(filename).stem
        if dataset_name.endswith(".tar"):
            dataset_name = dataset_name[:-4]
        return dataset_name or "custom-dataset", dataset
    available = ", ".join(available_dataset_presets())
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Use one of: {available}, pass a direct https URL, or pass a local .zip path."
    )


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    raw_data_root: Path
    processed_data_root: Path
    metadata_root: Path
    artifacts_root: Path
    mlruns_root: Path
    dataset_name: str
    dataset_url: str
    catalog_source: str
    catalog_limit: int | None
    catalog_min_votes: int
    catalog_include_adult: bool
    processed_version: str
    metadata_path: Path
    bundle_path: Path
    eval_report_path: Path
    feedback_log_path: Path
    download_ca_bundle: Path | None
    allow_insecure_download: bool
    default_sentence_model: str
    default_transformer_model: str
    default_ollama_model: str
    default_top_k: int = 10


def get_settings(
    dataset: str | None = None,
    catalog_source: str | None = None,
    catalog_limit: int | None = None,
    catalog_min_votes: int | None = None,
    catalog_include_adult: bool | None = None,
    ca_bundle_path: str | None = None,
    allow_insecure_download: bool | None = None,
) -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data"
    raw_data_root = data_root / "raw"
    processed_data_root = data_root / "processed"
    metadata_root = data_root / "metadata"
    artifacts_root = project_root / "artifacts"
    mlruns_root = project_root / "mlruns"
    dataset_name, dataset_url = resolve_dataset(dataset)
    resolved_catalog_source = (catalog_source or os.getenv("MOVIE_RECOMMENDER_CATALOG_SOURCE", "imdb")).strip().lower()
    resolved_catalog_limit = catalog_limit
    if resolved_catalog_limit is None:
        raw_limit = os.getenv("MOVIE_RECOMMENDER_CATALOG_LIMIT", "250000").strip()
        resolved_catalog_limit = None if raw_limit in {"", "0", "none", "null"} else int(raw_limit)
    resolved_catalog_min_votes = (
        catalog_min_votes if catalog_min_votes is not None else int(os.getenv("MOVIE_RECOMMENDER_CATALOG_MIN_VOTES", "25"))
    )
    resolved_catalog_include_adult = (
        catalog_include_adult
        if catalog_include_adult is not None
        else _env_flag("MOVIE_RECOMMENDER_CATALOG_INCLUDE_ADULT", default=False)
    )
    ca_bundle_value = ca_bundle_path or os.getenv("MOVIE_RECOMMENDER_CA_BUNDLE")
    ca_bundle = Path(ca_bundle_value).expanduser() if ca_bundle_value else None
    insecure_download = (
        allow_insecure_download
        if allow_insecure_download is not None
        else _env_flag("MOVIE_RECOMMENDER_INSECURE_DOWNLOAD", default=False)
    )

    return Settings(
        project_root=project_root,
        data_root=data_root,
        raw_data_root=raw_data_root,
        processed_data_root=processed_data_root,
        metadata_root=metadata_root,
        artifacts_root=artifacts_root,
        mlruns_root=mlruns_root,
        dataset_name=dataset_name,
        dataset_url=dataset_url,
        catalog_source=resolved_catalog_source,
        catalog_limit=resolved_catalog_limit,
        catalog_min_votes=resolved_catalog_min_votes,
        catalog_include_adult=resolved_catalog_include_adult,
        processed_version="v1",
        metadata_path=metadata_root / "sample_movie_metadata.json",
        bundle_path=artifacts_root / "model_bundle.pkl",
        eval_report_path=artifacts_root / "evaluation_report.json",
        feedback_log_path=artifacts_root / "feedback.jsonl",
        download_ca_bundle=ca_bundle,
        allow_insecure_download=insecure_download,
        default_sentence_model="sentence-transformers/all-MiniLM-L6-v2",
        default_transformer_model="google/flan-t5-small",
        default_ollama_model="llama3",
    )
