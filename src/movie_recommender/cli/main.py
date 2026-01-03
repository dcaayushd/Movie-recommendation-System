from __future__ import annotations

import argparse
import json
import subprocess
import sys

import uvicorn

from movie_recommender.config.settings import available_dataset_presets, get_settings
from movie_recommender.data.preprocess import load_processed_bundle, prepare_dataset
from movie_recommender.llm.factory import build_llm_backend
from movie_recommender.services.bundle import ModelBundle
from movie_recommender.services.evaluation import chronological_split, evaluate_bundle
from movie_recommender.services.training import TrainingConfig, train_model_bundle
from movie_recommender.utils.logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie recommendation system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_help = (
        "Dataset preset or direct HTTPS zip URL. "
        f"Built-in presets: {', '.join(available_dataset_presets())}."
    )

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--generate-summaries", action="store_true")
    prepare_parser.add_argument("--llm-backend", default="noop")
    prepare_parser.add_argument("--dataset", default=None, help=dataset_help)
    prepare_parser.add_argument("--catalog-source", default=None, help="Catalog source to expand the movie universe. Use `imdb` or `none`.")
    prepare_parser.add_argument("--catalog-limit", type=int, default=None, help="Max external catalog titles to import. Use 0 for no limit.")
    prepare_parser.add_argument("--catalog-min-votes", type=int, default=None, help="Minimum vote count for imported catalog titles.")
    prepare_parser.add_argument("--ca-bundle", default=None, help="Path to a trusted CA certificate bundle for HTTPS downloads.")
    prepare_parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="Disable TLS verification for dataset downloads on a trusted network.",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--llm-backend", default="noop")
    train_parser.add_argument("--disable-semantic", action="store_true")
    train_parser.add_argument("--dataset", default=None, help=dataset_help)
    train_parser.add_argument("--catalog-source", default=None, help="Catalog source to expand the movie universe. Use `imdb` or `none`.")
    train_parser.add_argument("--catalog-limit", type=int, default=None, help="Max external catalog titles to import. Use 0 for no limit.")
    train_parser.add_argument("--catalog-min-votes", type=int, default=None, help="Minimum vote count for imported catalog titles.")
    train_parser.add_argument("--ca-bundle", default=None, help="Path to a trusted CA certificate bundle for dataset downloads.")
    train_parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="Disable TLS verification for dataset downloads on a trusted network.",
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--top-k", type=int, default=10)

    serve_api_parser = subparsers.add_parser("serve-api")
    serve_api_parser.add_argument("--host", default="127.0.0.1")
    serve_api_parser.add_argument("--port", type=int, default=8000)

    serve_ui_parser = subparsers.add_parser("serve-ui")
    serve_ui_parser.add_argument("--port", type=int, default=8501)

    return parser


def main() -> None:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    settings = get_settings(
        dataset=getattr(args, "dataset", None),
        catalog_source=getattr(args, "catalog_source", None),
        catalog_limit=(None if getattr(args, "catalog_limit", None) == 0 else getattr(args, "catalog_limit", None)),
        catalog_min_votes=getattr(args, "catalog_min_votes", None),
        ca_bundle_path=getattr(args, "ca_bundle", None),
        allow_insecure_download=getattr(args, "insecure_download", None),
    )

    if args.command == "prepare-data":
        llm_backend = build_llm_backend(settings, args.llm_backend)
        prepare_dataset(
            settings,
            force_download=args.force,
            llm_backend=llm_backend,
            generate_summaries=args.generate_summaries,
        )
        return

    if args.command == "train":
        llm_backend = build_llm_backend(settings, args.llm_backend)
        prepared = prepare_dataset(settings, llm_backend=llm_backend, generate_summaries=False)
        config = TrainingConfig(use_semantic_embeddings=not args.disable_semantic)
        train_model_bundle(settings, prepared, training_config=config)
        return

    if args.command == "evaluate":
        bundle = ModelBundle.load(settings.bundle_path)
        split = chronological_split(bundle.ratings)
        report = evaluate_bundle(bundle, split, top_k=args.top_k)
        settings.eval_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if args.command == "serve-api":
        uvicorn.run("movie_recommender.api.app:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "serve-ui":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "apps/streamlit_app.py", "--server.port", str(args.port)],
            check=False,
        )
        return


if __name__ == "__main__":
    main()
