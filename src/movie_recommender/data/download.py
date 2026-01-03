from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from requests.exceptions import RequestException, SSLError

from movie_recommender.config.settings import Settings
from movie_recommender.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def _verify_option(settings: Settings) -> bool | str:
    if settings.allow_insecure_download:
        return False
    if settings.download_ca_bundle is not None:
        return str(settings.download_ca_bundle)
    return True


def _download_with_requests(url: str, archive_path: Path, verify: bool | str) -> None:
    with requests.get(url, stream=True, timeout=(20, 300), verify=verify) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_with_curl(url: str, archive_path: Path, settings: Settings) -> None:
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        "20",
        "--output",
        str(archive_path),
    ]
    if settings.download_ca_bundle is not None:
        command.extend(["--cacert", str(settings.download_ca_bundle)])
    if settings.allow_insecure_download:
        command.append("--insecure")
    command.append(url)
    subprocess.run(command, check=True)


def _download_archive(settings: Settings, archive_path: Path) -> None:
    if settings.dataset_url.startswith("file://"):
        parsed = urlparse(settings.dataset_url)
        source_path = Path(unquote(parsed.path))
        shutil.copy2(source_path, archive_path)
        return

    verify = _verify_option(settings)
    LOGGER.info(
        "Downloading open movie catalog",
        extra={
            "context": {
                "dataset": settings.dataset_name,
                "url": settings.dataset_url,
                "ca_bundle": str(settings.download_ca_bundle) if settings.download_ca_bundle else None,
                "insecure_download": settings.allow_insecure_download,
            }
        },
    )

    try:
        _download_with_requests(settings.dataset_url, archive_path, verify=verify)
        return
    except SSLError as exc:
        LOGGER.warning(
            "Requests download failed SSL verification; retrying with curl",
            extra={"context": {"error": str(exc), "dataset": settings.dataset_name}},
        )
    except RequestException as exc:
        LOGGER.warning(
            "Requests download failed; retrying with curl",
            extra={"context": {"error": str(exc), "dataset": settings.dataset_name}},
        )

    try:
        _download_with_curl(settings.dataset_url, archive_path, settings)
    except FileNotFoundError as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Dataset download failed and curl is not available. "
            "Install curl or rerun with MOVIE_RECOMMENDER_CA_BUNDLE pointing to your trusted CA certificate."
        ) from exc
    except subprocess.CalledProcessError as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Dataset download failed. If your network uses a custom CA, rerun with "
            "`movie-recommender prepare-data --ca-bundle /path/to/certificate.pem`. "
            "As a last resort on a trusted network, use `--insecure-download`, or pass a local catalog zip path with "
            "`--dataset /path/to/starter.zip`."
        ) from exc


def _extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    ensure_dir(extract_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    return _resolve_dataset_dir(extract_dir)


def _resolve_dataset_dir(extract_dir: Path) -> Path:
    nested_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(nested_dirs) == 1 and not (extract_dir / "movies.csv").exists():
        return nested_dirs[0]
    return extract_dir


def download_movielens_dataset(settings: Settings, force: bool = False) -> Path:
    ensure_dir(settings.raw_data_root)
    archive_path = settings.raw_data_root / f"{settings.dataset_name}.zip"
    extract_dir = settings.raw_data_root / settings.dataset_name

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if force and archive_path.exists():
        archive_path.unlink()

    if extract_dir.exists():
        LOGGER.info("Using existing raw dataset", extra={"context": {"path": str(extract_dir)}})
        return _resolve_dataset_dir(extract_dir)

    if not archive_path.exists():
        _download_archive(settings, archive_path)
    else:
        LOGGER.info("Using existing downloaded archive", extra={"context": {"path": str(archive_path)}})

    return _extract_archive(archive_path, extract_dir)
