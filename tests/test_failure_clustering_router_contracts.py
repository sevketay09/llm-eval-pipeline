"""Contract tests for api/routers/failure_clustering.py."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.failure_clustering import router as fc_router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(fc_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _report(low_scores=True):
    score = 0.3 if low_scores else 0.9
    return {
        "models": {
            "gpt-4o": {
                "tests": {
                    "qa": {
                        "results": [
                            {"case_id": f"c{i}", "question": f"question {i}", "scores": {"overall_score": score}, "category": "support"}
                            for i in range(6)
                        ]
                    }
                }
            }
        }
    }


class TestFailureClusteringEndpoint:
    def test_cluster_200(self, client):
        r = client.post("/api/failure-clustering", json={"report": _report()})
        assert r.status_code == 200

    def test_no_failures_returns_empty_clusters(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report(low_scores=False)}).json()
        assert body["total_failures"] == 0
        assert body["clusters"] == []

    def test_failures_detected(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        assert body["total_failures"] == 6

    def test_clusters_created(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        assert len(body["clusters"]) >= 2

    def test_response_schema(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        assert "total_failures" in body
        assert "threshold" in body
        assert "clusters" in body
        assert "model_breakdown" in body
        assert "category_breakdown" in body

    def test_cluster_schema(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        c = body["clusters"][0]
        for key in ("cluster_id", "size", "label", "centroid_text", "avg_score", "members"):
            assert key in c

    def test_custom_threshold(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report(low_scores=False), "threshold": 0.95}).json()
        assert body["total_failures"] == 6
        assert body["threshold"] == 0.95

    def test_model_breakdown(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        assert "gpt-4o" in body["model_breakdown"]

    def test_category_breakdown(self, client):
        body = client.post("/api/failure-clustering", json={"report": _report()}).json()
        assert "support" in body["category_breakdown"]

    def test_invalid_threshold_422(self, client):
        r = client.post("/api/failure-clustering", json={"report": _report(), "threshold": 1.5})
        assert r.status_code == 422
