from __future__ import annotations

from dataclasses import dataclass, field

from movie_recommender.llm.base import LLMBackend, fallback_explanation, fallback_summary, heuristic_parse_query


@dataclass
class TransformersBackend(LLMBackend):
    model_name: str
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def _load_pipeline(self) -> object:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        return self._pipeline

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        generator = self._load_pipeline()
        outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if not outputs:
            return ""
        return str(outputs[0].get("generated_text", "")).strip()

    def parse_query(self, query: str) -> dict:
        prompt = (
            "Return only JSON with keys seed_movie_title, genres, mood, time_of_day, top_k. "
            "Use null for missing values.\n"
            f"Query: {query}"
        )
        try:
            raw = self._generate(prompt, max_new_tokens=96)
            parsed = self.parse_json_payload(raw)
            return {**heuristic_parse_query(query), **{key: value for key, value in parsed.items() if value}}
        except Exception:
            return heuristic_parse_query(query)

    def summarize_movie(self, title: str, genres: str, overview: str, tags: str) -> str:
        prompt = (
            "Summarize this movie for a recommender system in 2 sentences.\n"
            f"Title: {title}\nGenres: {genres}\nOverview: {overview}\nTags: {tags}"
        )
        try:
            result = self._generate(prompt, max_new_tokens=80)
            return result or fallback_summary(title, genres, overview, tags)
        except Exception:
            return fallback_summary(title, genres, overview, tags)

    def explain_recommendation(self, title: str, reasons: list[str], score_breakdown: dict[str, float]) -> str:
        prompt = (
            "Explain why this movie was recommended in 2 sentences.\n"
            f"Movie: {title}\nReasons: {reasons}\nScores: {score_breakdown}"
        )
        try:
            result = self._generate(prompt, max_new_tokens=80)
            return result or fallback_explanation(title, reasons, score_breakdown)
        except Exception:
            return fallback_explanation(title, reasons, score_breakdown)

