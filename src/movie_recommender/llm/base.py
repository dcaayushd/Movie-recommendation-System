from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


KNOWN_GENRES = {
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film-noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
}
KNOWN_MOODS = {"uplifting", "dark", "funny", "romantic", "thoughtful", "intense", "relaxed", "family"}
KNOWN_TIMES = {"morning", "afternoon", "evening", "night"}
GENRE_ALIASES = {
    "kids": "children",
    "kid": "children",
    "children": "children",
    "family": "children",
    "scifi": "sci-fi",
    "science fiction": "sci-fi",
    "sci fi": "sci-fi",
    "romcom": "romance",
    "rom-com": "romance",
    "docs": "documentary",
    "documentaries": "documentary",
    "scary": "horror",
    "spooky": "horror",
    "suspense": "thriller",
}
MOOD_ALIASES = {
    "feel good": "uplifting",
    "feel-good": "uplifting",
    "lighthearted": "uplifting",
    "cozy": "relaxed",
    "chill": "relaxed",
    "easy": "relaxed",
    "date night": "romantic",
    "heartwarming": "uplifting",
    "smart": "thoughtful",
    "serious": "thoughtful",
    "exciting": "intense",
}
TIME_ALIASES = {
    "tonight": "night",
    "late night": "night",
    "this evening": "evening",
    "after work": "evening",
    "today afternoon": "afternoon",
}
SIGNAL_LABELS = {
    "popularity": "strong audience ratings",
    "quality": "excellent audience ratings",
    "content": "similar themes and genres",
    "audience": "audience review signals",
    "svd": "patterns from similar viewers",
    "matrix_factorization": "patterns from similar viewers",
    "autoencoder": "patterns from similar viewers",
    "rerank": "your current choices",
}


def _first_alias_match(query: str, aliases: dict[str, str]) -> str | None:
    for raw_value, normalized in aliases.items():
        if raw_value in query:
            return normalized
    return None


def _extract_seed_title(query: str) -> str | None:
    quoted = re.findall(r'"([^"]+)"', query)
    if not quoted:
        quoted = re.findall(r"'([^']+)'", query)
    if quoted:
        return quoted[0].strip()

    patterns = [
        r"(?:like|similar to|something like|based on|more like|same vibe as|in the style of)\s+([A-Za-z0-9:'().,\- ]+)",
        r"(?:movies? such as|movies? like)\s+([A-Za-z0-9:'().,\- ]+)",
    ]
    stop_words = {
        "for",
        "with",
        "about",
        "that",
        "and",
        "but",
        "please",
        "tonight",
        "today",
    }

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip().rstrip("?").rstrip(".")
        tokens = candidate.split()
        cleaned: list[str] = []
        for token in tokens:
            if token.lower() in stop_words:
                break
            cleaned.append(token)
        seed_title = " ".join(cleaned).strip()
        if seed_title:
            return seed_title
    return None


def _extract_top_k(query: str) -> int | None:
    patterns = [
        r"\btop\s+(\d{1,2})\b",
        r"\b(?:show|give|find|list)\s+me\s+(\d{1,2})\b",
        r"\b(?:recommend|suggest)\s+(\d{1,2})\s+(?:movies?|recommendations?|picks?|titles?)\b",
        r"\b(?:want|need)\s+(\d{1,2})\s+(?:movies?|recommendations?|picks?|titles?)\b",
        r"\b(\d{1,2})\s+(?:movies?|recommendations?|picks?|titles?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def heuristic_parse_query(query: str) -> dict:
    lowered = query.lower()
    seed_title = _extract_seed_title(query)

    genres = {genre for genre in KNOWN_GENRES if genre in lowered}
    alias_genre = _first_alias_match(lowered, GENRE_ALIASES)
    if alias_genre:
        genres.add(alias_genre)

    moods = [mood for mood in KNOWN_MOODS if mood in lowered]
    alias_mood = _first_alias_match(lowered, MOOD_ALIASES)
    if alias_mood and alias_mood not in moods:
        moods.append(alias_mood)

    times = [time_value for time_value in KNOWN_TIMES if time_value in lowered]
    alias_time = _first_alias_match(lowered, TIME_ALIASES)
    if alias_time and alias_time not in times:
        times.append(alias_time)

    return {
        "seed_movie_title": seed_title,
        "genres": sorted(genres),
        "mood": moods[0] if moods else None,
        "time_of_day": times[0] if times else None,
        "top_k": _extract_top_k(query),
    }


def fallback_summary(title: str, genres: str, overview: str, tags: str) -> str:
    parts = [title]
    if genres:
        parts.append(f"is a {genres.replace('|', ', ')} movie")
    if overview:
        parts.append(overview)
    elif tags:
        parts.append(f"Associated themes include {tags}.")
    return ". ".join(part.strip() for part in parts if part).strip()


def fallback_explanation(title: str, reasons: list[str], score_breakdown: dict[str, float]) -> str:
    dominant = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
    driver_labels: list[str] = []
    for name, value in dominant:
        if value <= 0:
            continue
        label = SIGNAL_LABELS.get(name)
        if label and label not in driver_labels:
            driver_labels.append(label)
        if len(driver_labels) == 2:
            break
    drivers = ", ".join(driver_labels)
    reason_text = "; ".join(reason for reason in reasons if reason) or "it matches your current taste profile"
    if drivers:
        return f"{title} fits because {reason_text}. It stood out for {drivers}."
    return f"{title} is recommended because {reason_text}."


class LLMBackend(ABC):
    @abstractmethod
    def parse_query(self, query: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def summarize_movie(self, title: str, genres: str, overview: str, tags: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def explain_recommendation(self, title: str, reasons: list[str], score_breakdown: dict[str, float]) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_json_payload(raw_text: str) -> dict:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output")
        return json.loads(raw_text[start : end + 1])


class NoOpLLMBackend(LLMBackend):
    def parse_query(self, query: str) -> dict:
        return heuristic_parse_query(query)

    def summarize_movie(self, title: str, genres: str, overview: str, tags: str) -> str:
        return fallback_summary(title, genres, overview, tags)

    def explain_recommendation(self, title: str, reasons: list[str], score_breakdown: dict[str, float]) -> str:
        return fallback_explanation(title, reasons, score_breakdown)
