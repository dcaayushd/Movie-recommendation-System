from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POSITIVE_RATING_THRESHOLD = 4.0
NEGATIVE_RATING_THRESHOLD = 2.5
AUDIENCE_MAX_FEATURES = 3000


def _normalize_rating_value(avg_rating: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(avg_rating, errors="coerce").fillna(0.0).astype(float)
    inferred_scale = np.where(numeric > 5.0, 10.0, 5.0)
    normalized = numeric / inferred_scale
    return pd.Series(np.clip(normalized, 0.0, 1.0), index=avg_rating.index, dtype=float)


def _safe_text(series: pd.Series | None, default: str = "", index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype="object") if index is not None else pd.Series(dtype="object")
    return series.fillna(default).astype(str)


def _token_count(text: str) -> int:
    tokens = {token.strip().lower() for token in str(text).replace("|", " ").split() if token.strip()}
    return len(tokens)


def _descriptor_text(row: pd.Series) -> str:
    parts: list[str] = []
    sentiment = float(row.get("audience_sentiment_score", 0.0))
    confidence = float(row.get("audience_confidence_score", 0.0))
    avg_rating = float(row.get("avg_rating", 0.0))

    if sentiment >= 0.55 and confidence >= 0.2:
        parts.append("audience loved it")
    elif sentiment >= 0.25:
        parts.append("audience recommended it")
    elif sentiment <= -0.2 and confidence >= 0.2:
        parts.append("audience found it divisive")

    if confidence >= 0.8:
        parts.append("widely watched")
    elif confidence >= 0.45:
        parts.append("well known")

    if avg_rating >= 4.2:
        parts.append("excellent ratings")
    elif avg_rating >= 3.8:
        parts.append("strong ratings")
    elif avg_rating >= 3.3:
        parts.append("solid ratings")

    return " ".join(parts).strip()


def enrich_movie_audience_signals(movies: pd.DataFrame, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = movies.copy()
    frame["movie_tag_text"] = _safe_text(frame.get("movie_tag_text"), "", index=frame.index)
    frame["overview"] = _safe_text(frame.get("overview"), "", index=frame.index)
    frame["genres"] = _safe_text(frame.get("genres"), "(no genres listed)", index=frame.index)
    frame["clean_title"] = _safe_text(frame.get("clean_title"), "Unknown", index=frame.index)
    avg_rating = frame.get("avg_rating", pd.Series(0.0, index=frame.index))
    rating_count = frame.get("rating_count", pd.Series(0, index=frame.index))
    frame["avg_rating"] = pd.to_numeric(avg_rating, errors="coerce").fillna(0.0).astype(float)
    frame["rating_count"] = pd.to_numeric(rating_count, errors="coerce").fillna(0).astype(int)

    if ratings is not None and not ratings.empty:
        rating_stats = (
            ratings.groupby("movie_id")["rating"]
            .agg(
                audience_positive_ratio=lambda values: float((values >= POSITIVE_RATING_THRESHOLD).mean()),
                audience_negative_ratio=lambda values: float((values <= NEGATIVE_RATING_THRESHOLD).mean()),
            )
            .reset_index()
        )
        frame = frame.drop(columns=["audience_positive_ratio", "audience_negative_ratio"], errors="ignore")
        frame = frame.merge(rating_stats, on="movie_id", how="left")

    rating_signal = _normalize_rating_value(frame["avg_rating"])
    estimated_positive_ratio = np.clip((rating_signal - 0.45) / 0.55, 0.0, 1.0)
    estimated_negative_ratio = np.clip((0.62 - rating_signal) / 0.62, 0.0, 1.0)
    positive_ratio = frame.get("audience_positive_ratio", pd.Series(np.nan, index=frame.index))
    negative_ratio = frame.get("audience_negative_ratio", pd.Series(np.nan, index=frame.index))
    frame["audience_positive_ratio"] = pd.to_numeric(positive_ratio, errors="coerce").fillna(estimated_positive_ratio)
    frame["audience_negative_ratio"] = pd.to_numeric(negative_ratio, errors="coerce").fillna(estimated_negative_ratio)

    max_votes = max(int(frame["rating_count"].max()), 1)
    frame["audience_confidence_score"] = np.log1p(frame["rating_count"].astype(float)) / np.log1p(float(max_votes))
    frame["audience_sentiment_score"] = (
        0.7 * frame["audience_positive_ratio"].astype(float) - 0.45 * frame["audience_negative_ratio"].astype(float)
    ).clip(-1.0, 1.0)
    frame["audience_tag_count"] = frame["movie_tag_text"].map(_token_count).astype(int)
    frame["audience_specificity_score"] = np.clip(frame["audience_tag_count"].astype(float) / 8.0, 0.0, 1.0)
    frame["audience_consensus_score"] = (
        0.42 * rating_signal
        + 0.26 * frame["audience_positive_ratio"].astype(float)
        + 0.17 * (1.0 - frame["audience_negative_ratio"].astype(float))
        + 0.15 * frame["audience_confidence_score"].astype(float)
    ).clip(0.0, 1.0)

    descriptor_text = frame.apply(_descriptor_text, axis=1)
    audience_basis = frame["movie_tag_text"].where(frame["movie_tag_text"].str.strip() != "", frame["overview"])
    audience_basis = audience_basis.where(audience_basis.str.strip() != "", frame["genres"].str.replace("|", " ", regex=False))
    frame["audience_review_text"] = (
        audience_basis.fillna("")
        + " "
        + frame["genres"].str.replace("|", " ", regex=False).fillna("")
        + " "
        + descriptor_text.fillna("")
    ).str.strip()
    return frame


@dataclass
class AudienceFeatureStore:
    movie_ids: list[int]
    movie_id_to_index: dict[int, int]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    consensus_scores: np.ndarray
    specificity_scores: np.ndarray

    def indices_for_movies(self, movie_ids: list[int]) -> list[int]:
        return [self.movie_id_to_index[movie_id] for movie_id in movie_ids if movie_id in self.movie_id_to_index]

    def vector_for_movie(self, movie_id: int):
        return self.tfidf_matrix[self.movie_id_to_index[movie_id]]


def fit_audience_features(movies: pd.DataFrame) -> AudienceFeatureStore:
    text_corpus = _safe_text(movies.get("audience_review_text"), "", index=movies.index).tolist()
    if not any(text.strip() for text in text_corpus):
        text_corpus = (
            _safe_text(movies.get("clean_title"), "", index=movies.index)
            + " "
            + _safe_text(movies.get("movie_tag_text"), "", index=movies.index)
            + " "
            + _safe_text(movies.get("genres"), "", index=movies.index).str.replace("|", " ", regex=False)
        ).str.strip().tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=AUDIENCE_MAX_FEATURES, ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)

    movie_ids = movies["movie_id"].astype(int).tolist()
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    return AudienceFeatureStore(
        movie_ids=movie_ids,
        movie_id_to_index=movie_id_to_index,
        tfidf_vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        consensus_scores=movies["audience_consensus_score"].fillna(0.0).astype(float).to_numpy(),
        specificity_scores=movies["audience_specificity_score"].fillna(0.0).astype(float).to_numpy(),
    )


@dataclass
class AudienceRecommender:
    store: AudienceFeatureStore
    movies: pd.DataFrame

    @classmethod
    def from_movies(cls, movies: pd.DataFrame) -> "AudienceRecommender":
        return cls(store=fit_audience_features(movies), movies=movies)

    def score_candidates(
        self,
        seed_movie_id: int | None = None,
        liked_movie_ids: list[int] | None = None,
        disliked_movie_ids: list[int] | None = None,
        candidate_movie_ids: list[int] | None = None,
        query_text: str | None = None,
    ) -> dict[int, float]:
        liked_movie_ids = liked_movie_ids or []
        disliked_movie_ids = disliked_movie_ids or []
        candidate_movie_ids = candidate_movie_ids or self.store.movie_ids
        indices = self.store.indices_for_movies(candidate_movie_ids)
        if not indices:
            return {}

        candidate_matrix = self.store.tfidf_matrix[indices]
        query_scores = self._query_scores(candidate_movie_ids, candidate_matrix, query_text)
        profile_scores = self._profile_scores(candidate_movie_ids, candidate_matrix, seed_movie_id, liked_movie_ids, disliked_movie_ids)
        consensus = np.array([self.store.consensus_scores[index] for index in indices], dtype=float)
        specificity = np.array([self.store.specificity_scores[index] for index in indices], dtype=float)

        if query_scores is not None and profile_scores is not None:
            final_scores = 0.38 * query_scores + 0.34 * profile_scores + 0.20 * consensus + 0.08 * specificity
        elif query_scores is not None:
            final_scores = 0.50 * query_scores + 0.34 * consensus + 0.16 * specificity
        elif profile_scores is not None:
            final_scores = 0.50 * profile_scores + 0.34 * consensus + 0.16 * specificity
        else:
            final_scores = 0.88 * consensus + 0.12 * specificity

        return {candidate_movie_ids[position]: float(final_scores[position]) for position in range(len(indices))}

    def highlight_terms(self, movie_id: int, limit: int = 3) -> list[str]:
        row = self.movies.loc[self.movies["movie_id"] == movie_id]
        if row.empty:
            return []
        raw_text = str(row.iloc[0].get("movie_tag_text", "")).replace("|", " ")
        tokens = []
        for token in raw_text.split():
            cleaned = token.strip().lower()
            if len(cleaned) < 4 or cleaned in tokens:
                continue
            tokens.append(cleaned)
            if len(tokens) == limit:
                break
        return tokens

    def _query_scores(self, candidate_movie_ids: list[int], candidate_matrix: sparse.csr_matrix, query_text: str | None) -> np.ndarray | None:
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return None
        query_vector = self.store.tfidf_vectorizer.transform([cleaned_query])
        return cosine_similarity(query_vector, candidate_matrix)[0]

    def _profile_scores(
        self,
        candidate_movie_ids: list[int],
        candidate_matrix: sparse.csr_matrix,
        seed_movie_id: int | None,
        liked_movie_ids: list[int],
        disliked_movie_ids: list[int],
    ) -> np.ndarray | None:
        profile_parts: list[sparse.csr_matrix] = []
        if seed_movie_id is not None and seed_movie_id in self.store.movie_id_to_index:
            profile_parts.append(self.store.vector_for_movie(seed_movie_id))
        liked_indices = self.store.indices_for_movies(liked_movie_ids)
        if liked_indices:
            profile_parts.append(self.store.tfidf_matrix[liked_indices].mean(axis=0))
        disliked_indices = self.store.indices_for_movies(disliked_movie_ids)
        if disliked_indices:
            profile_parts.append(-0.35 * self.store.tfidf_matrix[disliked_indices].mean(axis=0))
        if not profile_parts:
            return None
        dense_parts = [sparse.csr_matrix(part) if not sparse.issparse(part) else part.tocsr() for part in profile_parts]
        profile_vector = dense_parts[0].copy()
        for part in dense_parts[1:]:
            profile_vector = profile_vector + part
        profile_vector = profile_vector / len(dense_parts)
        return cosine_similarity(profile_vector, candidate_matrix)[0]
