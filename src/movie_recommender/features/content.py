from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)

CONTENT_QUERY_MAX_FEATURES = 5000
CONTENT_QUERY_EXPANSIONS = {
    "smart": "cerebral intelligent thoughtful mind-bending",
    "intelligent": "smart cerebral thoughtful",
    "emotional": "heartfelt moving poignant human",
    "emotional depth": "heartfelt moving poignant human",
    "audience reviews": "audience recommended it audience loved it strong ratings excellent ratings well known",
    "strong audience reviews": "audience loved it strong ratings excellent ratings well known",
    "highly rated": "excellent ratings strong ratings widely watched",
    "best": "excellent ratings audience loved it",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _safe_text(series: pd.Series | None, default: str = "", index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype="object") if index is not None else pd.Series(dtype="object")
    return series.fillna(default).astype(str)


def _build_query_corpus(movies: pd.DataFrame) -> list[str]:
    index = movies.index
    genres = _safe_text(movies.get("genres"), "", index=index).str.replace("|", " ", regex=False)
    tags = _safe_text(movies.get("movie_tag_text"), "", index=index)
    audience = _safe_text(movies.get("audience_review_text"), "", index=index)
    overview = _safe_text(movies.get("overview"), "", index=index)
    summaries = _safe_text(movies.get("llm_summary"), "", index=index)
    query_corpus = (genres + " " + tags + " " + audience + " " + overview + " " + summaries).str.strip()

    if any(text.strip() for text in query_corpus.tolist()):
        return query_corpus.tolist()

    fallback = _safe_text(movies.get("combined_text"), "", index=index).str.strip()
    if any(text.strip() for text in fallback.tolist()):
        return fallback.tolist()

    return (_safe_text(movies.get("clean_title"), "", index=index) + " " + genres).str.strip().tolist()


def _expand_query_text(query_text: str) -> str:
    cleaned = (query_text or "").strip().lower()
    if not cleaned:
        return ""
    expansions = [cleaned]
    for phrase, related_terms in CONTENT_QUERY_EXPANSIONS.items():
        if phrase in cleaned:
            expansions.append(related_terms)
    if "sci-fi" in cleaned or "science fiction" in cleaned or "scifi" in cleaned:
        expansions.append("science fiction futuristic cerebral")
    return " ".join(dict.fromkeys(expansions))


def _tokenize_text(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall((text or "").lower()) if token}


@dataclass
class ContentFeatureStore:
    movie_ids: list[int]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    semantic_embeddings: np.ndarray | None
    semantic_model_name: str | None
    movie_id_to_index: dict[int, int]
    query_tfidf_vectorizer: TfidfVectorizer | None = None
    query_tfidf_matrix: sparse.csr_matrix | None = None
    clean_titles: list[str] | None = None

    def vector_for_movie(self, movie_id: int) -> np.ndarray:
        index = self.movie_id_to_index[movie_id]
        if self.semantic_embeddings is not None:
            return self.semantic_embeddings[index]
        return self.tfidf_matrix[index].toarray()[0]

    def indices_for_movies(self, movie_ids: list[int]) -> list[int]:
        return [self.movie_id_to_index[movie_id] for movie_id in movie_ids if movie_id in self.movie_id_to_index]


def fit_content_features(
    movies: pd.DataFrame,
    sentence_model_name: str,
    use_semantic: bool = True,
) -> ContentFeatureStore:
    text_corpus = movies["combined_text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=CONTENT_QUERY_MAX_FEATURES, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    query_vectorizer = TfidfVectorizer(stop_words="english", max_features=CONTENT_QUERY_MAX_FEATURES, ngram_range=(1, 2))
    query_tfidf_matrix = query_vectorizer.fit_transform(_build_query_corpus(movies))

    semantic_embeddings = None
    semantic_model_name: str | None = None
    if use_semantic:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(sentence_model_name)
            semantic_embeddings = model.encode(text_corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            semantic_model_name = sentence_model_name
        except Exception as exc:
            LOGGER.warning("Semantic embeddings unavailable; falling back to TF-IDF only", extra={"context": {"error": str(exc)}})

    movie_ids = movies["movie_id"].astype(int).tolist()
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    return ContentFeatureStore(
        movie_ids=movie_ids,
        tfidf_vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        semantic_embeddings=semantic_embeddings,
        semantic_model_name=semantic_model_name,
        movie_id_to_index=movie_id_to_index,
        query_tfidf_vectorizer=query_vectorizer,
        query_tfidf_matrix=query_tfidf_matrix,
        clean_titles=movies["clean_title"].fillna("").astype(str).tolist() if "clean_title" in movies.columns else None,
    )


def ensure_query_content_features(store: ContentFeatureStore, movies: pd.DataFrame) -> ContentFeatureStore:
    query_vectorizer = getattr(store, "query_tfidf_vectorizer", None)
    query_tfidf_matrix = getattr(store, "query_tfidf_matrix", None)
    if query_vectorizer is not None and query_tfidf_matrix is not None and query_tfidf_matrix.shape[0] == len(store.movie_ids):
        return store

    if movies.empty:
        store.query_tfidf_vectorizer = store.tfidf_vectorizer
        store.query_tfidf_matrix = store.tfidf_matrix
        return store

    movie_frame = movies.drop_duplicates(subset=["movie_id"]).set_index("movie_id")
    aligned_movies = movie_frame.reindex(store.movie_ids).reset_index()
    query_vectorizer = TfidfVectorizer(stop_words="english", max_features=CONTENT_QUERY_MAX_FEATURES, ngram_range=(1, 2))
    query_tfidf_matrix = query_vectorizer.fit_transform(_build_query_corpus(aligned_movies))
    store.query_tfidf_vectorizer = query_vectorizer
    store.query_tfidf_matrix = query_tfidf_matrix
    if "clean_title" in aligned_movies.columns:
        store.clean_titles = aligned_movies["clean_title"].fillna("").astype(str).tolist()
    return store


def profile_embedding(store: ContentFeatureStore, movie_ids: list[int]) -> np.ndarray:
    valid_ids = [movie_id for movie_id in movie_ids if movie_id in store.movie_id_to_index]
    if not valid_ids:
        width = store.semantic_embeddings.shape[1] if store.semantic_embeddings is not None else store.tfidf_matrix.shape[1]
        return np.zeros(width, dtype=float)
    vectors = np.vstack([store.vector_for_movie(movie_id) for movie_id in valid_ids])
    return vectors.mean(axis=0)


def cosine_scores_from_profile(
    store: ContentFeatureStore,
    profile: np.ndarray,
    candidate_movie_ids: list[int] | None = None,
) -> dict[int, float]:
    if profile.ndim == 1:
        profile = profile.reshape(1, -1)
    candidate_movie_ids = candidate_movie_ids or store.movie_ids
    indices = store.indices_for_movies(candidate_movie_ids)
    if not indices:
        return {}
    if store.semantic_embeddings is not None:
        candidate_matrix = store.semantic_embeddings[indices]
        scores = cosine_similarity(profile, candidate_matrix)[0]
    else:
        candidate_matrix = store.tfidf_matrix[indices]
        scores = cosine_similarity(profile, candidate_matrix)[0]
    return {candidate_movie_ids[position]: float(scores[position]) for position in range(len(indices))}


def cosine_scores_from_text_query(
    store: ContentFeatureStore,
    query_text: str,
    candidate_movie_ids: list[int] | None = None,
) -> dict[int, float]:
    cleaned_query = (query_text or "").strip()
    if not cleaned_query:
        return {}
    candidate_movie_ids = candidate_movie_ids or store.movie_ids
    indices = store.indices_for_movies(candidate_movie_ids)
    if not indices:
        return {}
    expanded_query = _expand_query_text(cleaned_query)
    query_vectorizer = getattr(store, "query_tfidf_vectorizer", None)
    if query_vectorizer is None:
        query_vectorizer = store.tfidf_vectorizer
    query_tfidf_matrix = getattr(store, "query_tfidf_matrix", None)
    if query_tfidf_matrix is None:
        query_tfidf_matrix = store.tfidf_matrix
    query_vector = query_vectorizer.transform([expanded_query])
    candidate_matrix = query_tfidf_matrix[indices]
    if query_vector.nnz == 0 and query_vectorizer is not store.tfidf_vectorizer:
        query_vector = store.tfidf_vectorizer.transform([cleaned_query])
        candidate_matrix = store.tfidf_matrix[indices]
    scores = cosine_similarity(query_vector, candidate_matrix)[0]
    if len(cleaned_query.split()) <= 4:
        title_scores = cosine_similarity(store.tfidf_vectorizer.transform([cleaned_query]), store.tfidf_matrix[indices])[0]
        scores = np.maximum(scores, title_scores)
        clean_titles = getattr(store, "clean_titles", None)
        query_tokens = _tokenize_text(cleaned_query)
        if clean_titles and query_tokens:
            overlap_scores = np.array(
                [
                    float(len(query_tokens & _tokenize_text(clean_titles[index])) / len(query_tokens))
                    for index in indices
                ],
                dtype=float,
            )
            scores = np.maximum(scores, overlap_scores)
    return {candidate_movie_ids[position]: float(scores[position]) for position in range(len(indices))}


def pair_similarity(store: ContentFeatureStore, movie_id_a: int, movie_id_b: int) -> float:
    if movie_id_a not in store.movie_id_to_index or movie_id_b not in store.movie_id_to_index:
        return 0.0
    if store.semantic_embeddings is not None:
        vector_a = store.semantic_embeddings[store.movie_id_to_index[movie_id_a]]
        vector_b = store.semantic_embeddings[store.movie_id_to_index[movie_id_b]]
        return float(np.dot(vector_a, vector_b))
    matrix_a = store.tfidf_matrix[store.movie_id_to_index[movie_id_a]]
    matrix_b = store.tfidf_matrix[store.movie_id_to_index[movie_id_b]]
    return float(cosine_similarity(matrix_a, matrix_b)[0][0])
