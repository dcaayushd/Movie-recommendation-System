from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

from movie_recommender.llm.base import LLMBackend, fallback_explanation, fallback_summary, heuristic_parse_query


@dataclass
class OllamaBackend(LLMBackend):
    model_name: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 60

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return str(data.get("response", "")).strip()

    def parse_query(self, query: str) -> dict:
        prompt = (
            "Extract a JSON object with keys seed_movie_title, genres, mood, time_of_day, top_k from this query. "
            "Use null when unknown.\n"
            f"Query: {query}"
        )
        try:
            raw = self._generate(prompt)
            parsed = self.parse_json_payload(raw)
            return {**heuristic_parse_query(query), **{key: value for key, value in parsed.items() if value}}
        except Exception:
            return heuristic_parse_query(query)

    def summarize_movie(self, title: str, genres: str, overview: str, tags: str) -> str:
        prompt = (
            "Write a compact recommendation-oriented movie summary in 2 sentences or fewer.\n"
            f"Title: {title}\nGenres: {genres}\nOverview: {overview}\nTags: {tags}"
        )
        try:
            return self._generate(prompt)
        except Exception:
            return fallback_summary(title, genres, overview, tags)

    def explain_recommendation(self, title: str, reasons: list[str], score_breakdown: dict[str, float]) -> str:
        prompt = (
            "Explain this movie recommendation in 2 sentences or fewer.\n"
            f"Movie: {title}\nReasons: {reasons}\nScore breakdown: {score_breakdown}"
        )
        try:
            return self._generate(prompt)
        except Exception:
            return fallback_explanation(title, reasons, score_breakdown)

