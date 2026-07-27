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


class TestEvaluateWithJudgeModel:
    """judge_model wiring — real adapter factory injection, mirroring how
    RagEvalService's embedding_adapter_factory is tested."""

    def _fake_adapter_factory(self, model_key, config_path):
        if model_key == "does-not-exist":
            raise ValueError(f"Model '{model_key}' not found in config")

        class _FakeAdapter:
            def generate(self, messages):
                return {"content": '{"score": 0.9, "reasoning": "matches criteria"}'}

        return _FakeAdapter()

    def _failing_adapter_factory(self, model_key, config_path):
        class _FailingAdapter:
            def generate(self, messages):
                return {"content": None, "error": "upstream 500"}

        return _FailingAdapter()

    def _client_with(self, adapter_factory):
        app = FastAPI()
        svc = CustomMetricService(adapter_factory=adapter_factory)
        app.dependency_overrides[get_service] = lambda: svc
        app.include_router(custom_metrics_router, prefix="/api")
        return TestClient(app)

    def test_real_judge_model_produces_score(self):
        client = self._client_with(self._fake_adapter_factory)
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}],
            "judge_model": "deepseek-v4-flash",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["score"] == 0.9
        assert body["avg_score"] == 0.9

    def test_unknown_judge_model_returns_404(self):
        client = self._client_with(self._fake_adapter_factory)
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}],
            "judge_model": "does-not-exist",
        })
        assert r.status_code == 404

    def test_judge_model_generation_failure_becomes_per_case_error_not_a_fake_score(self):
        client = self._client_with(self._failing_adapter_factory)
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}],
            "judge_model": "deepseek-v4-flash",
        })
        assert r.status_code == 200
        result = r.json()["results"][0]
        assert result["score"] is None
        assert "upstream 500" in result["error"]

    def test_no_judge_model_still_dry_runs(self):
        client = self._client_with(self._fake_adapter_factory)
        mid = client.post("/api/custom-metrics", json=_create_payload()).json()["metric_id"]
        r = client.post(f"/api/custom-metrics/{mid}/evaluate", json={
            "cases": [{"question": "Q", "answer": "A"}],
        })
        body = r.json()
        assert body["results"][0]["reasoning"] == "No model configured."


class TestCustomMetricServicePersistence:
    def test_save_and_load_from_round_trips(self, tmp_path):
        path = tmp_path / "custom_metrics.json"
        svc = CustomMetricService()
        rec = svc.create(name="Empathy", description="Rate empathy")
        svc.save(path)

        reloaded = CustomMetricService()
        reloaded.load_from(path)

        restored = reloaded.get(rec.metric_id)
        assert restored is not None
        assert restored.name == "Empathy"
        assert restored.prompt == rec.prompt
        assert restored.status == "ready"

    def test_load_from_missing_file_is_noop(self, tmp_path):
        svc = CustomMetricService()
        svc.load_from(tmp_path / "does-not-exist.json")
        assert svc.list() == []
