# Deployment Guide

## Local Development

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e ".[dev]"
$ movie-recommender prepare-data --dataset starter
$ movie-recommender train --dataset starter
$ movie-recommender serve-api
$ movie-recommender serve-ui
```

## Catalog Presets

The CLI accepts these free presets:

- `starter`
- `expanded`
- `benchmark`
- `classic`

You can also pass a direct HTTPS zip URL or a local `.zip` catalog path with `--dataset`.

## Large Catalog Mode

The default setup now expands the title catalog with IMDb non-commercial title files while keeping the interaction dataset for personalization.

Useful variants:

```console
$ movie-recommender prepare-data --dataset starter --catalog-source none
$ movie-recommender prepare-data --dataset starter --catalog-source imdb --catalog-limit 0
```

## SSL / Proxy Troubleshooting

If Python download verification fails because your network injects a custom certificate:

```console
$ movie-recommender prepare-data --ca-bundle /path/to/certificate.pem
```

Or set:

```console
$ export MOVIE_RECOMMENDER_CA_BUNDLE=/path/to/certificate.pem
```

The downloader will automatically retry with `curl` after Python SSL failures. If you are on a trusted internal network and still need a bypass, use `--insecure-download`.

You can also bypass network issues entirely by downloading the archive yourself and pointing the CLI to it:

```console
$ movie-recommender prepare-data --dataset ~/Downloads/starter.zip
```

## Optional Ollama Setup

```console
$ ollama serve
$ ollama pull llama3
```

Set `MOVIE_RECOMMENDER_LLM_BACKEND=ollama` to enable the Ollama adapter.

## Docker Compose

```console
$ docker compose up --build
```

The API will start on `http://localhost:8000` and the Streamlit app on `http://localhost:8501`.
