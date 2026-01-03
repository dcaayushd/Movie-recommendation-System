from __future__ import annotations

import os

from movie_recommender.config.settings import Settings
from movie_recommender.llm.base import LLMBackend, NoOpLLMBackend
from movie_recommender.llm.ollama_backend import OllamaBackend
from movie_recommender.llm.transformers_backend import TransformersBackend


def build_llm_backend(settings: Settings, backend_name: str | None = None) -> LLMBackend:
    backend_name = backend_name or os.getenv("MOVIE_RECOMMENDER_LLM_BACKEND", "noop")
    backend_name = backend_name.lower()

    if backend_name == "ollama":
        return OllamaBackend(model_name=os.getenv("MOVIE_RECOMMENDER_OLLAMA_MODEL", settings.default_ollama_model))
    if backend_name == "transformers":
        return TransformersBackend(model_name=os.getenv("MOVIE_RECOMMENDER_TRANSFORMERS_MODEL", settings.default_transformer_model))
    return NoOpLLMBackend()
