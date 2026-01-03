from __future__ import annotations

import pandas as pd

from movie_recommender.features.content import cosine_scores_from_text_query, fit_content_features


def test_query_scoring_prefers_descriptive_match_over_title_keyword_leakage():
    movies = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "clean_title": "Get Smart, Again!",
                "genres": "Comedy|Sci-Fi",
                "movie_tag_text": "",
                "audience_review_text": "Comedy Sci-Fi well known",
                "overview": "",
                "llm_summary": "",
                "combined_text": "Get Smart Again Comedy Sci-Fi Comedy Sci-Fi well known",
            },
            {
                "movie_id": 2,
                "clean_title": "Arrival",
                "genres": "Sci-Fi|Drama",
                "movie_tag_text": "cerebral emotional thoughtful intelligent",
                "audience_review_text": "cerebral emotional thoughtful audience loved it strong ratings",
                "overview": "A cerebral and emotional first-contact drama.",
                "llm_summary": "",
                "combined_text": "Arrival Sci-Fi Drama cerebral emotional thoughtful intelligent audience loved it strong ratings",
            },
        ]
    )

    store = fit_content_features(movies, sentence_model_name="unused", use_semantic=False)

    scores = cosine_scores_from_text_query(store, "smart sci-fi with strong audience reviews and emotional depth")

    assert scores[2] > scores[1]


def test_query_scoring_falls_back_to_title_match_for_title_only_search():
    movies = pd.DataFrame(
        [
            {
                "movie_id": 1,
                "clean_title": "Get Smart, Again!",
                "genres": "Comedy|Sci-Fi",
                "movie_tag_text": "",
                "audience_review_text": "Comedy Sci-Fi well known",
                "overview": "",
                "llm_summary": "",
                "combined_text": "Get Smart Again Comedy Sci-Fi Comedy Sci-Fi well known",
            },
            {
                "movie_id": 2,
                "clean_title": "Arrival",
                "genres": "Sci-Fi|Drama",
                "movie_tag_text": "cerebral emotional thoughtful intelligent",
                "audience_review_text": "cerebral emotional thoughtful audience loved it strong ratings",
                "overview": "A cerebral and emotional first-contact drama.",
                "llm_summary": "",
                "combined_text": "Arrival Sci-Fi Drama cerebral emotional thoughtful intelligent audience loved it strong ratings",
            },
        ]
    )

    store = fit_content_features(movies, sentence_model_name="unused", use_semantic=False)

    scores = cosine_scores_from_text_query(store, "Get Smart Again")

    assert scores[1] > scores[2]
