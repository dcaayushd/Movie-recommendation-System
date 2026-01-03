from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st

from movie_recommender.config.settings import get_settings
from movie_recommender.llm.factory import build_llm_backend
from movie_recommender.services.inference import MovieRecommenderService, RecommendationRequest

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
GENERIC_QUERY_HINTS = {
    "",
    "recommend like this",
    "recommend me like this",
    "recommend on the basis of this",
    "recommend me on the basis of this",
    "recommend me",
}
EXAMPLE_PROMPTS = [
    ("Funny tonight", "Something funny for tonight"),
    ("Like Interstellar", "Something like Interstellar but easier to watch"),
    ("Family movie", "A good family movie for kids"),
    ("Smart drama", "A thoughtful drama with strong characters"),
]
SEARCH_STOPWORDS = {"a", "an", "the", "movie", "film"}
TITLE_YEAR_PATTERN = re.compile(r"\s*\((19\d{2}|20\d{2})\)\s*$")
QUERY_YEAR_PATTERN = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MovieSearchEntry:
    label: str
    lower_label: str
    normalized_label: str
    title: str
    normalized_title: str
    meaningful_tokens: frozenset[str]
    year: int | None
    popularity_rank: int


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f8f5ef;
        }
        [data-testid="stHeader"] {
            background: rgba(248, 245, 239, 0.96);
        }
        .block-container {
            max-width: 980px;
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
        }
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none;
        }
        h1, h2, h3, h4, h5, h6,
        p, label, [data-testid="stMarkdownContainer"] {
            color: #111827;
        }
        .hero {
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #111827;
            line-height: 1.05;
            margin: 0;
        }
        .hero-copy {
            margin-top: 0.45rem;
            color: #6b7280;
            font-size: 0.98rem;
        }
        .soft-note {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin: 0.65rem 0;
            font-size: 0.92rem;
        }
        .result-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.18rem;
        }
        .result-meta {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 0.7rem;
        }
        .result-body {
            color: #374151;
            line-height: 1.6;
            margin-bottom: 0.55rem;
        }
        .result-subtle {
            color: #6b7280;
            font-size: 0.85rem;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea,
        input {
            background: #ffffff !important;
            color: #111827 !important;
            border-color: #d6d3d1 !important;
        }
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span {
            color: #111827 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[role="listbox"] {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #e7e5e4 !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08) !important;
        }
        div[role="option"] {
            background: #ffffff !important;
            color: #111827 !important;
        }
        div[role="option"]:hover {
            background: #fef2f2 !important;
        }
        div[data-baseweb="tag"] {
            background: #fee2e2 !important;
            border-color: #fecaca !important;
            color: #991b1b !important;
        }
        button {
            border-radius: 12px !important;
            font-weight: 700 !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        button[kind="primary"] {
            background: #ef4444 !important;
            color: #ffffff !important;
            border: 1px solid #ef4444 !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="secondary"],
        button[kind="secondary"] {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d6d3d1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bundle_signature(settings) -> tuple[float, float]:
    app_path = Path(__file__)
    bundle_path = settings.bundle_path
    return (
        bundle_path.stat().st_mtime if bundle_path.exists() else 0.0,
        app_path.stat().st_mtime,
    )


@st.cache_resource(show_spinner=False)
def get_local_service(signature: tuple[float, float]) -> MovieRecommenderService | None:
    settings = get_settings()
    if not settings.bundle_path.exists():
        return None
    llm_backend = build_llm_backend(settings)
    return MovieRecommenderService.from_path(settings, llm_backend=llm_backend)


@st.cache_data(ttl=4, show_spinner=False)
def fetch_api_status(api_base_url: str) -> dict | None:
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/status", timeout=(1.2, 2.0))
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def initialize_state() -> None:
    st.session_state.setdefault("prompt_input", "")
    st.session_state.setdefault("prompt_base_text", "")
    st.session_state.setdefault("seed_movie_input", None)
    st.session_state.setdefault("seed_movie_label", "")
    st.session_state.setdefault("mood_value", "")
    st.session_state.setdefault("time_value", "")
    st.session_state.setdefault("selected_genres", [])
    st.session_state.setdefault("top_k_value", 8)
    st.session_state.setdefault("last_auto_prompt", "")
    st.session_state.setdefault("auto_refresh_requested", False)
    st.session_state.setdefault("liked_movie_ids", [])
    st.session_state.setdefault("disliked_movie_ids", [])
    st.session_state.setdefault("latest_results", [])
    st.session_state.setdefault("resolved_query", None)
    st.session_state.setdefault("notice", None)


def clear_runtime_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def reset_state() -> None:
    st.session_state["prompt_input"] = ""
    st.session_state["prompt_base_text"] = ""
    st.session_state["seed_movie_input"] = None
    st.session_state["seed_movie_label"] = ""
    st.session_state["mood_value"] = ""
    st.session_state["time_value"] = ""
    st.session_state["selected_genres"] = []
    st.session_state["top_k_value"] = 8
    st.session_state["last_auto_prompt"] = ""
    st.session_state["auto_refresh_requested"] = False
    st.session_state["liked_movie_ids"] = []
    st.session_state["disliked_movie_ids"] = []
    st.session_state["latest_results"] = []
    st.session_state["resolved_query"] = None
    st.session_state["notice"] = None


def movie_id_options(service: MovieRecommenderService | None) -> dict[str, int]:
    if service is None:
        return {}
    options: dict[str, int] = {}
    ranked_movies = service.movies.sort_values(["rating_count", "avg_rating", "clean_title"], ascending=[False, False, True])
    for row in ranked_movies.itertuples(index=False):
        year = getattr(row, "year", None)
        has_year = year is not None and year == year
        label = f"{row.clean_title} ({int(year)})" if has_year else str(row.clean_title)
        options[label] = int(row.movie_id)
    return options


def available_genres(service: MovieRecommenderService | None) -> list[str]:
    if service is None:
        return []
    return sorted({genre for value in service.movies["genres"] for genre in str(value).split("|") if genre})


def normalize_search_text(text: str | None) -> str:
    lowered = (text or "").strip().lower()
    normalized = NON_ALNUM_PATTERN.sub(" ", lowered)
    return " ".join(normalized.split())


def split_label_title_and_year(label: str) -> tuple[str, int | None]:
    match = TITLE_YEAR_PATTERN.search(label)
    if not match:
        return label.strip(), None
    return label[: match.start()].strip(), int(match.group(1))


def tokenize_search_text(text: str | None) -> frozenset[str]:
    tokens = [token for token in normalize_search_text(text).split() if len(token) > 1]
    meaningful_tokens = [token for token in tokens if token not in SEARCH_STOPWORDS]
    return frozenset(meaningful_tokens or tokens)


def normalized_query_title(raw_text: str) -> str:
    without_year = QUERY_YEAR_PATTERN.sub(" ", raw_text or "")
    return normalize_search_text(without_year)


def extract_query_year(raw_text: str) -> int | None:
    match = QUERY_YEAR_PATTERN.search(raw_text or "")
    return int(match.group(1)) if match else None


@lru_cache(maxsize=8)
def movie_search_entries(labels: tuple[str, ...]) -> tuple[MovieSearchEntry, ...]:
    entries: list[MovieSearchEntry] = []
    for popularity_rank, label in enumerate(labels):
        title, year = split_label_title_and_year(label)
        entries.append(
            MovieSearchEntry(
                label=label,
                lower_label=label.lower(),
                normalized_label=normalize_search_text(label),
                title=title,
                normalized_title=normalize_search_text(title),
                meaningful_tokens=tokenize_search_text(title),
                year=year,
                popularity_rank=popularity_rank,
            )
        )
    return tuple(entries)


def popularity_prior(rank: int) -> float:
    return max(0.0, 6.0 - rank * 0.05)


def score_movie_candidate(
    entry: MovieSearchEntry,
    raw_query: str,
    normalized_query: str,
    query_title: str,
    query_tokens: frozenset[str],
    query_year: int | None,
) -> float:
    raw_query_lower = raw_query.lower()
    if raw_query_lower == entry.lower_label:
        return 220.0 + popularity_prior(entry.popularity_rank)
    if normalized_query == entry.normalized_label:
        return 210.0 + popularity_prior(entry.popularity_rank)
    if raw_query_lower == entry.title.lower():
        score = 190.0
        if query_year is not None:
            score += 28.0 if entry.year == query_year else -28.0
        return score + popularity_prior(entry.popularity_rank)
    if query_title and query_title == entry.normalized_title:
        score = 180.0
        if query_year is not None:
            score += 28.0 if entry.year == query_year else -28.0
        return score + popularity_prior(entry.popularity_rank)

    score = 0.0
    if raw_query_lower and entry.lower_label.startswith(raw_query_lower):
        score += 42.0
    if query_title and entry.normalized_title.startswith(query_title):
        score += 38.0
    if raw_query_lower and raw_query_lower in entry.lower_label:
        score += min(26.0, 8.0 + len(raw_query_lower) * 1.25)
    if query_title and query_title in entry.normalized_title:
        score += min(32.0, 10.0 + len(query_title) * 1.4)

    if query_tokens:
        overlap = len(query_tokens & entry.meaningful_tokens)
        if overlap:
            recall = overlap / len(query_tokens)
            precision = overlap / len(entry.meaningful_tokens) if entry.meaningful_tokens else 0.0
            score += 42.0 * recall + 12.0 * precision
            if overlap == len(query_tokens):
                score += 12.0

    fuzzy_query = query_title or normalized_query
    if fuzzy_query:
        score += 26.0 * SequenceMatcher(None, fuzzy_query, entry.normalized_title).ratio()
        score += 8.0 * SequenceMatcher(None, normalized_query, entry.normalized_label).ratio()

    if query_year is not None:
        if entry.year == query_year:
            score += 24.0
        elif entry.year is not None:
            score -= min(12.0, abs(entry.year - query_year) / 4.0)

    return score + popularity_prior(entry.popularity_rank)


def ranked_movie_labels(options: dict[str, int], search_text: str, limit: int = 30) -> list[tuple[str, float]]:
    labels = tuple(options.keys())
    if not labels:
        return []

    query = (search_text or "").strip()
    if not query:
        return [(label, popularity_prior(index)) for index, label in enumerate(labels[:limit])]

    normalized_query = normalize_search_text(query)
    query_title = normalized_query_title(query)
    query_tokens = tokenize_search_text(query_title or query)
    query_year = extract_query_year(query)
    scored = [
        (
            entry.label,
            score_movie_candidate(entry, query, normalized_query, query_title, query_tokens, query_year),
        )
        for entry in movie_search_entries(labels)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def filter_movie_labels(options: dict[str, int], search_text: str, selected_label: str, limit: int = 30) -> list[str]:
    matches = [label for label, _score in ranked_movie_labels(options, search_text, limit=limit)]
    if selected_label and selected_label not in matches and selected_label in options:
        matches = [selected_label] + matches[: max(limit - 1, 0)]
    return matches


def should_auto_resolve(search_text: str, ranked_matches: list[tuple[str, float]]) -> bool:
    if not ranked_matches:
        return False
    query_title = normalized_query_title(search_text)
    compact_query = (query_title or normalize_search_text(search_text)).replace(" ", "")
    best_score = ranked_matches[0][1]
    second_score = ranked_matches[1][1] if len(ranked_matches) > 1 else float("-inf")
    if len(compact_query) < 4:
        return False
    return best_score >= 90.0 or (best_score >= 72.0 and best_score - second_score >= 8.0)


def resolve_movie_label(options: dict[str, int], raw_text: str) -> tuple[str, list[str]]:
    typed_text = (raw_text or "").strip()
    if not typed_text:
        return "", []
    if typed_text in options:
        return typed_text, [typed_text]

    ranked_matches = ranked_movie_labels(options, typed_text, limit=5)
    matches = [label for label, _score in ranked_matches]
    if not ranked_matches:
        return "", []

    best_label = matches[0]
    best_title, _best_year = split_label_title_and_year(best_label)
    normalized_query = normalize_search_text(typed_text)
    query_title = normalized_query_title(typed_text)

    if normalized_query == normalize_search_text(best_label) or query_title == normalize_search_text(best_title):
        return best_label, matches
    if should_auto_resolve(typed_text, ranked_matches):
        return best_label, matches
    return "", matches


def sync_seed_from_picker(options: dict[str, int]) -> None:
    raw_value = st.session_state.get("seed_movie_input")
    typed_text = (raw_value or "").strip()
    if not typed_text:
        st.session_state["seed_movie_input"] = None
        st.session_state["seed_movie_label"] = ""
        sync_prompt_from_controls()
        return

    resolved_label, _matches = resolve_movie_label(options, typed_text)
    st.session_state["seed_movie_label"] = resolved_label
    if resolved_label and typed_text.lower() != resolved_label.lower():
        st.session_state["seed_movie_input"] = resolved_label
    sync_prompt_from_controls()


def normalize_results(results: list | None) -> list[dict]:
    normalized: list[dict] = []
    for result in results or []:
        normalized.append(result if isinstance(result, dict) else result.__dict__)
    return normalized


def feedback_payload(movie_id: int, sentiment: str) -> dict:
    return {
        "movie_id": int(movie_id),
        "sentiment": sentiment,
        "user_id": None,
        "source": "streamlit",
    }


def submit_feedback(runtime_mode: str, api_base_url: str, service: MovieRecommenderService | None, payload: dict) -> None:
    try:
        if runtime_mode == "api":
            requests.post(f"{api_base_url.rstrip('/')}/feedback", json=payload, timeout=(1.5, 4)).raise_for_status()
        elif service is not None:
            service.record_feedback(payload)
    except requests.RequestException:
        pass


def is_generic_query(query: str | None) -> bool:
    return (query or "").strip().lower() in GENERIC_QUERY_HINTS


def build_auto_prompt_preview(request: RecommendationRequest, seed_label: str) -> str:
    if request.query and not is_generic_query(request.query):
        return request.query.strip()

    parts = ["Recommend movies"]
    if seed_label:
        parts.append(f"similar to {seed_label}")
    if request.genres:
        parts.append(f"with {', '.join(request.genres[:3])}")
    if request.mood:
        parts.append(f"for a {request.mood} mood")
    if request.time_of_day:
        parts.append(f"for {request.time_of_day}")
    if not seed_label and not request.genres:
        parts.append("that are easy to enjoy and well liked")
    return " ".join(parts)


def build_prompt_from_base(base_text: str, seed_label: str, mood: str | None, time_of_day: str | None, genres: list[str]) -> str:
    base = (base_text or "").strip()
    if not base or is_generic_query(base):
        request = RecommendationRequest(
            user_id=None,
            seed_movie_id=None,
            liked_movie_ids=[],
            disliked_movie_ids=[],
            query=None,
            mood=mood,
            time_of_day=time_of_day,
            top_k=st.session_state.get("top_k_value", 8),
            genres=genres,
        )
        return build_auto_prompt_preview(request, seed_label)

    pieces = [base.rstrip(" .")]
    base_lower = base.lower()
    if seed_label and seed_label.lower() not in base_lower:
        pieces.append(f"like {seed_label}")
    if mood and mood.lower() not in base_lower:
        pieces.append(f"for a {mood} mood")
    if time_of_day and time_of_day.lower() not in base_lower:
        pieces.append(f"for {time_of_day}")
    missing_genres = [genre for genre in genres[:3] if genre.lower() not in base_lower]
    if missing_genres:
        pieces.append(f"with {', '.join(missing_genres)}")
    return " ".join(piece for piece in pieces if piece).strip()


def remember_prompt_base() -> None:
    current_prompt = (st.session_state.get("prompt_input") or "").strip()
    st.session_state["prompt_base_text"] = current_prompt
    st.session_state["last_auto_prompt"] = ""


def request_auto_refresh() -> None:
    if st.session_state.get("latest_results"):
        st.session_state["auto_refresh_requested"] = True


def sync_prompt_from_controls() -> None:
    current_prompt = (st.session_state.get("prompt_input") or "").strip()
    last_auto_prompt = (st.session_state.get("last_auto_prompt") or "").strip()
    seed_label = st.session_state.get("seed_movie_label", "")
    mood = st.session_state.get("mood_value") or None
    time_of_day = st.session_state.get("time_value") or None
    genres = list(st.session_state.get("selected_genres", []))
    base_prompt = (st.session_state.get("prompt_base_text") or "").strip()
    if not base_prompt and current_prompt and current_prompt != last_auto_prompt and not is_generic_query(current_prompt):
        base_prompt = current_prompt
        st.session_state["prompt_base_text"] = base_prompt
    updated_prompt = build_prompt_from_base(base_prompt, seed_label, mood, time_of_day, genres)
    st.session_state["prompt_input"] = updated_prompt
    st.session_state["last_auto_prompt"] = updated_prompt
    request_auto_refresh()


def call_api_recommend(api_base_url: str, request: RecommendationRequest) -> tuple[list[dict], str | None]:
    payload = {
        "user_id": request.user_id,
        "seed_movie_id": request.seed_movie_id,
        "liked_movie_ids": request.liked_movie_ids,
        "disliked_movie_ids": request.disliked_movie_ids,
        "query": request.query,
        "mood": request.mood,
        "time_of_day": request.time_of_day,
        "top_k": request.top_k,
        "genres": request.genres,
        "year_from": request.year_from,
        "year_to": request.year_to,
    }
    if request.query:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/chat",
            json={"query": request.query, "top_k": request.top_k},
            timeout=(1.5, 8),
        )
        response.raise_for_status()
        chat_payload = response.json()
        return normalize_results(chat_payload.get("results", [])), chat_payload.get("resolved_query")
    response = requests.post(f"{api_base_url.rstrip('/')}/recommend", json=payload, timeout=(1.5, 8))
    response.raise_for_status()
    return normalize_results(response.json()), None


def call_local_recommend(service: MovieRecommenderService, request: RecommendationRequest) -> tuple[list[dict], str | None]:
    if request.query:
        payload = service.chat(request.query, top_k=request.top_k)
        return normalize_results(payload["results"]), payload.get("resolved_query")
    return normalize_results(service.recommend(request)), None


def run_recommendation(
    runtime_mode: str,
    api_base_url: str,
    service: MovieRecommenderService | None,
    request: RecommendationRequest,
) -> tuple[list[dict], str | None, str | None]:
    if runtime_mode == "api":
        try:
            results, resolved_query = call_api_recommend(api_base_url, request)
            return results, resolved_query, None
        except requests.RequestException:
            if service is not None:
                results, resolved_query = call_local_recommend(service, request)
                return results, resolved_query, "The app was slow for a moment, so I kept going locally."
            return [], None, "Could not reach the recommender right now."
    if service is not None:
        results, resolved_query = call_local_recommend(service, request)
        return results, resolved_query, None
    return [], None, "No trained model found yet."


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Movie Recommender</div>
            <div class="hero-copy">Type what you feel like watching and get simple, useful picks.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_prompts() -> None:
    st.caption("Try asking like this")
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for index, (label, prompt) in enumerate(EXAMPLE_PROMPTS):
        if example_cols[index].button(label, key=f"example-{index}", width="stretch"):
            st.session_state["prompt_input"] = prompt
            st.session_state["prompt_base_text"] = prompt
            st.session_state["last_auto_prompt"] = ""
            st.session_state["notice"] = None
            st.rerun()


def render_controls(options: dict[str, int], genres: list[str]) -> tuple[bool, RecommendationRequest, str, str]:
    render_example_prompts()
    with st.container(border=True):
        query = st.text_area(
            "What do you want to watch?",
            key="prompt_input",
            placeholder='Examples: "something funny for tonight", "like Interstellar but lighter", "good family movie"',
            height=110,
            on_change=remember_prompt_base,
        )

        row = st.columns([2.1, 1, 1])
        with row[0]:
            seed_input = st.selectbox(
                "Movie you already like",
                list(options.keys()),
                index=None,
                key="seed_movie_input",
                placeholder="Type or pick a movie, like Interstellar",
                accept_new_options=True,
                on_change=sync_seed_from_picker,
                args=(options,),
            )
            resolved_seed_label, seed_matches = resolve_movie_label(options, seed_input or "")
            seed_movie_label = resolved_seed_label or st.session_state.get("seed_movie_label", "")

            if seed_input and seed_movie_label and str(seed_input).strip().lower() != seed_movie_label.lower():
                st.caption(f"Using: {seed_movie_label}")
            elif seed_input and not seed_movie_label and seed_matches:
                st.caption(f"Closest matches: {', '.join(seed_matches[:3])}")
            elif seed_input and not seed_movie_label:
                st.caption("No close matches yet. Try a different spelling or another title.")
                seed_movie_label = ""

        mood = row[1].selectbox(
            "Mood",
            ["", "uplifting", "funny", "thoughtful", "romantic", "intense", "relaxed", "dark", "family"],
            key="mood_value",
            on_change=sync_prompt_from_controls,
        )
        time_of_day = row[2].selectbox(
            "When",
            ["", "morning", "afternoon", "evening", "night"],
            key="time_value",
            on_change=sync_prompt_from_controls,
        )

        with st.expander("More options"):
            st.multiselect("Genres", genres, key="selected_genres", on_change=sync_prompt_from_controls)
            st.select_slider("How many picks?", options=[6, 8, 10, 12], key="top_k_value", on_change=request_auto_refresh)

        submitted = st.button("Find movies", width="stretch", type="primary")

    request = RecommendationRequest(
        user_id=None,
        seed_movie_id=options.get(seed_movie_label),
        liked_movie_ids=st.session_state["liked_movie_ids"],
        disliked_movie_ids=st.session_state["disliked_movie_ids"],
        query=query or None,
        mood=mood or None,
        time_of_day=time_of_day or None,
        top_k=st.session_state["top_k_value"],
        genres=list(st.session_state["selected_genres"]),
    )
    auto_prompt_preview = build_auto_prompt_preview(request, seed_movie_label)
    if is_generic_query(query):
        st.markdown(f"<div class='soft-note'>I’ll use: {auto_prompt_preview}</div>", unsafe_allow_html=True)
        st.session_state["last_auto_prompt"] = auto_prompt_preview
    return submitted, request, seed_movie_label, auto_prompt_preview


def render_results(results: list[dict], runtime_mode: str, api_base_url: str, service: MovieRecommenderService | None) -> None:
    st.markdown("### Picks")
    if not results:
        st.caption("Ask for something above and your recommendations will appear here.")
        return

    for result in results:
        with st.container(border=True):
            has_poster = bool(result.get("poster_url"))
            if has_poster:
                poster_col, body_col = st.columns([0.9, 2.6])
            else:
                poster_col = None
                body_col = st.container()

            if has_poster and poster_col is not None:
                with poster_col:
                    st.image(result["poster_url"])

            with body_col:
                st.markdown(f"<div class='result-title'>{result['title']}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='result-meta'>{str(result['genres']).replace('|', ' · ')} · Average rating {result['avg_rating']:.2f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div class='result-body'>{result['explanation']}</div>", unsafe_allow_html=True)

                overview = (result.get("overview") or "").strip()
                if overview:
                    short_overview = overview if len(overview) <= 180 else f"{overview[:177].rstrip()}..."
                    st.markdown(f"<div class='result-subtle'>{short_overview}</div>", unsafe_allow_html=True)

                button_cols = st.columns(2)
                if button_cols[0].button("Like this", key=f"like-{result['movie_id']}", width="stretch"):
                    if result["movie_id"] not in st.session_state["liked_movie_ids"]:
                        st.session_state["liked_movie_ids"].append(result["movie_id"])
                    submit_feedback(runtime_mode, api_base_url, service, feedback_payload(result["movie_id"], "like"))
                    st.rerun()
                if button_cols[1].button("Not for me", key=f"hide-{result['movie_id']}", width="stretch"):
                    if result["movie_id"] not in st.session_state["disliked_movie_ids"]:
                        st.session_state["disliked_movie_ids"].append(result["movie_id"])
                    submit_feedback(runtime_mode, api_base_url, service, feedback_payload(result["movie_id"], "dislike"))
                    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
    apply_theme()
    initialize_state()
    settings = get_settings()
    api_status = fetch_api_status(DEFAULT_API_BASE_URL)
    local_service = get_local_service(bundle_signature(settings))
    runtime_mode = "api" if api_status and api_status.get("bundle_ready") else ("local" if local_service is not None else "offline")

    render_header()

    if runtime_mode == "offline":
        st.warning("No trained movie model is ready yet. Run training first.")
        return

    notice = st.session_state.get("notice")
    if notice:
        st.markdown(f"<div class='soft-note'>{notice}</div>", unsafe_allow_html=True)

    actions = st.columns([1.4, 1])
    with actions[1]:
        if st.button("Start over", width="stretch"):
            reset_state()
            clear_runtime_caches()
            st.rerun()

    options = movie_id_options(local_service)
    genres = available_genres(local_service)
    submitted, request, seed_label, auto_prompt_preview = render_controls(options, genres)

    auto_refresh_requested = bool(st.session_state.get("auto_refresh_requested"))
    if submitted or auto_refresh_requested:
        with st.spinner("Finding movies for you..."):
            results, resolved_query, notice_text = run_recommendation(runtime_mode, DEFAULT_API_BASE_URL, local_service, request)
        st.session_state["latest_results"] = results
        st.session_state["resolved_query"] = resolved_query or auto_prompt_preview
        st.session_state["notice"] = notice_text
        st.session_state["auto_refresh_requested"] = False

    resolved_query = st.session_state.get("resolved_query")
    if resolved_query:
        if seed_label:
            st.caption(f"Based on {seed_label} and: {resolved_query}")
        else:
            st.caption(f"Based on: {resolved_query}")

    render_results(st.session_state.get("latest_results", []), runtime_mode, DEFAULT_API_BASE_URL, local_service)


if __name__ == "__main__":
    main()
