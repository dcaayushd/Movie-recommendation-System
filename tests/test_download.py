from __future__ import annotations

import zipfile
from pathlib import Path

from requests.exceptions import SSLError

from movie_recommender.config.settings import resolve_dataset
from movie_recommender.data.download import download_movielens_dataset


def _write_fake_archive(archive_path: Path, dataset_name: str) -> None:
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(f"{dataset_name}/movies.csv", "movieId,title,genres\n1,Toy Story (1995),Animation|Children|Comedy\n")
        zf.writestr(f"{dataset_name}/ratings.csv", "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n")
        zf.writestr(f"{dataset_name}/tags.csv", "userId,movieId,tag,timestamp\n1,1,pixar,964982703\n")


def test_resolve_dataset_supports_presets_and_direct_urls():
    starter_name, starter_url = resolve_dataset("starter")
    assert starter_name == "starter"
    assert starter_url.endswith("ml-latest-small.zip")

    name, url = resolve_dataset("ml-latest-small")
    assert name == "ml-latest-small"
    assert url.endswith("ml-latest-small.zip")

    custom_name, custom_url = resolve_dataset("https://example.com/free-datasets/movies.zip")
    assert custom_name == "movies"
    assert custom_url == "https://example.com/free-datasets/movies.zip"


def test_resolve_dataset_supports_local_archives(tmp_path):
    archive_path = tmp_path / "ml-latest-small.zip"
    archive_path.write_bytes(b"zip")
    name, url = resolve_dataset(str(archive_path))
    assert name == "ml-latest-small"
    assert url.startswith("file://")


def test_download_falls_back_to_curl_on_ssl_error(settings, monkeypatch):
    calls: list[str] = []

    def fake_requests(_url, _archive_path, verify):
        calls.append(f"requests:{verify}")
        raise SSLError("self-signed certificate in certificate chain")

    def fake_curl(_url, archive_path, download_settings):
        calls.append(f"curl:{download_settings.dataset_name}")
        _write_fake_archive(archive_path, download_settings.dataset_name)

    monkeypatch.setattr("movie_recommender.data.download._download_with_requests", fake_requests)
    monkeypatch.setattr("movie_recommender.data.download._download_with_curl", fake_curl)

    extracted_dir = download_movielens_dataset(settings, force=True)
    assert extracted_dir.exists()
    assert (extracted_dir / "movies.csv").exists()
    assert calls[0].startswith("requests:")
    assert calls[1] == f"curl:{settings.dataset_name}"

    repeated_dir = download_movielens_dataset(settings, force=False)
    assert repeated_dir == extracted_dir
