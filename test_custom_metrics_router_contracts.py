"""Contract tests for api/routers/custom_metrics.py."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.custom_metrics import router as custom_metrics_router, get_service
from api.services.custom_metric_service import CustomMetricService


@pytest.fixture()
def client():
    app = FastAPI()
    svc = CustomMetricService()
    app.dependency_overrides[get_service] = lambda: svc
    app.include_router(custom_metrics_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _create_payload(name="Empathy", description="Rate how empathetic the response is"):
    return {"name": name, "description": description}


class TestCreateEndpoint:
    def test_create_201(self, client):
        r = client.post("/api/custom-metrics", json=_create_payload())
        assert r.status_code == 201

    def test_create_returns_detail(self, client):
        r = client.post("/api/custom-metrics", json=_create_payload("My Metric"))
        body = r.json()
        assert body["name"] == "My Metric"
        assert body["status"] == "ready"
        assert "prompt" in body
        assert "metric_id" in body

    def test_create_generates_nonempty_prompt(self, client):
        r = client.post("/api/custom-metrics", json=_create_payload())
        body = r.json()
        assert len(body["prompt"]) > 10

    def test_create_requires_name(self, client):
        r = client.post("/api/custom-metrics", json={"description": "Rate stuff"})
        assert r.status_code == 422

    def test_create_requires_description(self, client):
        r = client.post("/api/custom-metrics", json={"name": "Metric"})
        assert r.status_code == 422


class TestListEndpoint:
    def test_list_empty(self, client):
        r = client.get("/api/custom-metrics")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_create(self, client):
        client.post("/api/custom-metrics", json=_create_payload())
        client.post("/api/custom-metrics", json=_create_payload("Metric 2", "desc"))
        r = client.get("/api/custom-metrics")
        assert len(r.json()) == 2


class TestGetEndpoint:
    def test_get_existing(self, client):
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.get(f"/api/custom-metrics/{mid}")
        assert r.status_code == 200
        assert r.json()["metric_id"] == mid

    def test_get_missing_404(self, client):
        r = client.get("/api/custom-metrics/nonexistent")
        assert r.status_code == 404


class TestEvaluateEndpoint:
    def test_evaluate_200(self, client):
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}]
        })
        assert r.status_code == 200

    def test_evaluate_missing_metric_404(self, client):
        r = client.post("/api/custom-metrics/nonexistent/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}]
        })
        assert r.status_code == 404

    def test_evaluate_returns_result_per_case(self, client):
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        cases = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(3)]
        body = client.post(f"/api/custom-metrics/{mid}/evaluate", json={"cases": cases}).json()
        assert len(body["results"]) == 3

    def test_evaluate_result_schema(self, client):
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        body = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}]
        }).json()
        r = body["results"][0]
        for key in ("question", "answer", "reasoning"):
            assert key in r

    def test_evaluate_requires_min_1_case(self, client):
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={"cases": []})
        assert r.status_code == 422
