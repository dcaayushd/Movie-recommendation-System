from __future__ import annotations

from movie_recommender.services.inference import RecommendationRequest


def test_service_recommendation_respects_exclusions(service):
    results = service.recommend(
        RecommendationRequest(
            user_id=1,
            seed_movie_id=1,
            disliked_movie_ids=[3],
            top_k=3,
        )
    )
    assert results
    assert all(result.movie_id not in {1, 3} for result in results)
    assert all(result.explanation for result in results)


def test_service_chat_and_similar(service):
    response = service.chat('Recommend me something romantic like "Sabrina"', top_k=2)
    assert response["results"]
    assert response["resolved_query"]
    similar = service.similar_movies(4, top_k=2)
    assert len(similar) == 2


def test_service_chat_auto_prompt_for_generic_query(service):
    response = service.chat("recommend me", top_k=2)
    assert len(response["results"]) == 2
    assert response["resolved_query"]
    assert response["resolved_query"].lower().startswith("recommend movies")


def test_service_chat_handles_simple_natural_language(service):
    response = service.chat("funny movie for kids tonight", top_k=3)
    assert response["results"]
    assert len(response["results"]) <= 3
    assert response["resolved_query"]


def test_service_recommend_returns_requested_result_count(service):
    results = service.recommend(RecommendationRequest(top_k=5))
    assert len(results) == 5


def test_service_quality_score_prefers_better_rated_titles(service):
    assert service._quality_score(2) > service._quality_score(4)
