from __future__ import annotations

from movie_recommender.ranking.audience import AudienceRecommender, enrich_movie_audience_signals


def test_enrich_movie_audience_signals_adds_consensus_columns(prepared_bundle):
    enriched = enrich_movie_audience_signals(prepared_bundle.movies, ratings=prepared_bundle.ratings)

    assert "audience_review_text" in enriched.columns
    assert "audience_consensus_score" in enriched.columns
    assert enriched["audience_consensus_score"].between(0.0, 1.0).all()


def test_audience_recommender_uses_audience_tags_for_matching(prepared_bundle):
    recommender = AudienceRecommender.from_movies(prepared_bundle.movies)
    candidate_ids = prepared_bundle.movies["movie_id"].astype(int).tolist()

    scores = recommender.score_candidates(query_text="spy thriller", candidate_movie_ids=candidate_ids)

    ranked_ids = [movie_id for movie_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    assert ranked_ids[0] == 5
