"""Contract tests for api/routers/experiments.py — TestClient."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.experiments import router as experiments_router, get_service
from api.services.experiment_service import ExperimentService
from experiments.store import ExperimentStore


def fake_model(system_prompt: str, user_input: str):
    return f"ok: {user_input[:20]}", 10.0


def _fake_adapter_factory(model_key, config_path):
    """Stands in for a real UnifiedLLMAdapter — "test-model" always resolves,
    mirroring how rag_eval/custom_metrics tests inject a fake adapter factory
    instead of hitting real config/models.yaml + a live provider."""
    if model_key == "does-not-exist":
        raise ValueError(f"Model '{model_key}' not found in config")

    class _FakeAdapter:
        def generate(self, messages):
            user_content = messages[-1]["content"]
            return {"content": f"ok: {user_content[:20]}", "latency": 0.01}

    return _FakeAdapter()


@pytest.fixture()
def client():
    app = FastAPI()
    svc = ExperimentService(store=ExperimentStore(), adapter_factory=_fake_adapter_factory)
    app.dependency_overrides[get_service] = lambda: svc
    app.include_router(experiments_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _create_payload(name="exp1", n_variants=2, n_cases=2):
    return {
        "name": name,
        "model_key": "test-model",
        "variants": [
            {"label": f"v{i}", "system_prompt": f"System prompt {i}"}
            for i in range(n_variants)
        ],
        "dataset": [
            {"case_id": f"c{i}", "input": f"question {i}", "expected": f"answer {i}"}
            for i in range(n_cases)
        ],
    }


class TestCreateEndpoint:
    def test_create_201(self, client):
        r = client.post("/api/experiments", json=_create_payload())
        assert r.status_code == 201

    def test_create_returns_summary(self, client):
        r = client.post("/api/experiments", json=_create_payload("my-exp"))
        body = r.json()
        assert body["name"] == "my-exp"
        assert body["status"] == "pending"
        assert body["variant_count"] == 2
        assert body["case_count"] == 2

    def test_create_requires_min_2_variants(self, client):
        payload = _create_payload()
        payload["variants"] = [{"label": "v1", "system_prompt": "only one"}]
        r = client.post("/api/experiments", json=payload)
        assert r.status_code == 422

    def test_create_requires_min_1_case(self, client):
        payload = _create_payload()
        payload["dataset"] = []
        r = client.post("/api/experiments", json=payload)
        assert r.status_code == 422


class TestListEndpoint:
    def test_list_empty(self, client):
        r = client.get("/api/experiments")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_create(self, client):
        client.post("/api/experiments", json=_create_payload())
        client.post("/api/experiments", json=_create_payload("exp2"))
        r = client.get("/api/experiments")
        assert len(r.json()) == 2


class TestGetEndpoint:
    def test_get_existing(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        r = client.get(f"/api/experiments/{eid}")
        assert r.status_code == 200
        assert r.json()["experiment_id"] == eid

    def test_get_missing_404(self, client):
        r = client.get("/api/experiments/nonexistent")
        assert r.status_code == 404

    def test_get_returns_variants_and_dataset(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        r = client.get(f"/api/experiments/{eid}")
        body = r.json()
        assert len(body["variants"]) == 2
        assert len(body["dataset"]) == 2


class TestRunEndpoint:
    def test_run_pending_experiment(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        r = client.post(f"/api/experiments/{eid}/run", json={})
        assert r.status_code == 202

    def test_run_missing_404(self, client):
        r = client.post("/api/experiments/nope/run", json={})
        assert r.status_code == 404

    def test_run_returns_done_status(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        r = client.post(f"/api/experiments/{eid}/run", json={})
        assert r.json()["status"] == "done"

    def test_run_actually_calls_the_configured_model_not_a_noop(self, client):
        """Regression: model_key was previously stored but never wired to a
        real adapter — run() always used the noop placeholder regardless of
        model_key. Assert the fake adapter's real output shows up."""
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        client.post(f"/api/experiments/{eid}/run", json={})
        detail = client.get(f"/api/experiments/{eid}").json()
        outputs = [r["output"] for r in detail["results"]]
        assert all(o.startswith("ok: ") for o in outputs)
        assert not any("no model configured" in o for o in outputs)

    def test_run_with_unknown_model_key_becomes_error_status_not_a_crash(self, client):
        payload = _create_payload()
        payload["model_key"] = "does-not-exist"
        cr = client.post("/api/experiments", json=payload)
        eid = cr.json()["experiment_id"]
        r = client.post(f"/api/experiments/{eid}/run", json={})
        assert r.status_code == 202
        assert r.json()["status"] == "error"
        detail = client.get(f"/api/experiments/{eid}").json()
        assert "does-not-exist" in detail["error"]

    def test_run_with_empty_model_key_still_dry_runs(self, client):
        payload = _create_payload()
        payload["model_key"] = ""
        cr = client.post("/api/experiments", json=payload)
        eid = cr.json()["experiment_id"]
        client.post(f"/api/experiments/{eid}/run", json={})
        detail = client.get(f"/api/experiments/{eid}").json()
        assert detail["status"] == "done"
        assert "no model configured" in detail["results"][0]["output"]


class TestCompareEndpoint:
    def test_compare_after_run(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        client.post(f"/api/experiments/{eid}/run", json={})
        r = client.get(f"/api/experiments/{eid}/compare")
        assert r.status_code == 200
        body = r.json()
        assert "diffs" in body
        assert body["experiment_id"] == eid

    def test_compare_not_done_409(self, client):
        cr = client.post("/api/experiments", json=_create_payload())
        eid = cr.json()["experiment_id"]
        r = client.get(f"/api/experiments/{eid}/compare")
        assert r.status_code == 409

    def test_compare_missing_404(self, client):
        r = client.get("/api/experiments/nope/compare")
        assert r.status_code == 404

    def test_compare_verdict_counts(self, client):
        cr = client.post("/api/experiments", json=_create_payload(n_cases=3))
        eid = cr.json()["experiment_id"]
        client.post(f"/api/experiments/{eid}/run", json={})
        r = client.get(f"/api/experiments/{eid}/compare")
        body = r.json()
        total = body["improved"] + body["regressed"] + body["stable"] + body["missing"]
        assert total == len(body["diffs"])
