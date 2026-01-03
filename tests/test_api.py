from __future__ import annotations

from fastapi.testclient import TestClient

from movie_recommender.api.app import create_app


def test_api_endpoints(settings, trained_bundle):
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["bundle_ready"] is True

        status = client.get("/status")
        assert status.status_code == 200
        assert status.json()["bundle_ready"] is True

        recommend = client.post("/recommend", json={"top_k": 3})
        assert recommend.status_code == 200
        assert len(recommend.json()) == 3

        similar = client.get("/similar/1", params={"top_k": 2})
        assert similar.status_code == 200
        assert len(similar.json()) == 2

        chat = client.post("/chat", json={"query": 'Recommend me movies like "Toy Story"', "top_k": 2})
        assert chat.status_code == 200
        assert "parsed_query" in chat.json()
        assert "resolved_query" in chat.json()

        feedback = client.post("/feedback", json={"movie_id": 1, "sentiment": "like"})
        assert feedback.status_code == 200
        assert settings.feedback_log_path.exists()

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "/recommend" in metrics.json()
