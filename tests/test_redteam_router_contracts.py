"""Contract tests for api/routers/redteam.py."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.redteam import router as redteam_router, get_service
from api.services.redteam_service import RedTeamService
from redteam.store import RedTeamStore


def _refusal_model(system_prompt: str, user_input: str):
    return "I cannot help with that.", 5.0


def _fake_adapter_factory(model_key, config_path):
    """Stands in for a real UnifiedLLMAdapter, mirroring the pattern used in
    experiments/custom_metrics tests instead of hitting real config/models.yaml."""
    if model_key == "does-not-exist":
        raise ValueError(f"Model '{model_key}' not found in config")

    class _FakeAdapter:
        def generate(self, messages):
            return {"content": "I cannot help with that.", "latency": 0.005}

    return _FakeAdapter()


@pytest.fixture()
def client():
    app = FastAPI()
    svc = RedTeamService(store=RedTeamStore())
    app.dependency_overrides[get_service] = lambda: svc
    app.include_router(redteam_router, prefix="/api")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_with_model():
    app = FastAPI()
    svc = RedTeamService(store=RedTeamStore(), adapter_factory=_fake_adapter_factory)
    app.dependency_overrides[get_service] = lambda: svc
    app.include_router(redteam_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _session_payload(categories=None):
    return {
        "system_prompt": "You are a helpful assistant.",
        "categories": categories or ["jailbreak", "prompt_injection"],
    }


class TestCreateEndpoint:
    def test_create_201(self, client):
        r = client.post("/api/redteam", json=_session_payload())
        assert r.status_code == 201

    def test_create_returns_summary(self, client):
        r = client.post("/api/redteam", json=_session_payload())
        body = r.json()
        assert body["status"] == "pending"
        assert body["attack_count"] > 0
        assert "session_id" in body

    def test_create_requires_system_prompt(self, client):
        r = client.post("/api/redteam", json={"categories": ["jailbreak"]})
        assert r.status_code == 422

    def test_create_empty_system_prompt_rejected(self, client):
        r = client.post("/api/redteam", json={"system_prompt": "", "categories": ["jailbreak"]})
        assert r.status_code == 422

    def test_create_default_categories(self, client):
        r = client.post("/api/redteam", json={"system_prompt": "be helpful"})
        assert r.status_code == 201
        body = r.json()
        assert body["attack_count"] > 0


class TestListEndpoint:
    def test_list_empty(self, client):
        r = client.get("/api/redteam")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_after_create(self, client):
        client.post("/api/redteam", json=_session_payload())
        client.post("/api/redteam", json=_session_payload())
        r = client.get("/api/redteam")
        assert len(r.json()) == 2


class TestGetEndpoint:
    def test_get_existing(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        r = client.get(f"/api/redteam/{sid}")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert "attacks" in body
        assert "results" in body

    def test_get_missing_404(self, client):
        r = client.get("/api/redteam/nonexistent")
        assert r.status_code == 404


class TestRunEndpoint:
    def test_run_202(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        r = client.post(f"/api/redteam/{sid}/run")
        assert r.status_code == 202

    def test_run_missing_404(self, client):
        r = client.post("/api/redteam/nonexistent/run")
        assert r.status_code == 404

    def test_run_sets_status_done(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        body = client.get(f"/api/redteam/{sid}").json()
        assert body["status"] == "done"

    def test_run_populates_results(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        body = client.get(f"/api/redteam/{sid}").json()
        assert len(body["results"]) > 0

    def test_run_without_model_key_uses_noop_not_a_real_model(self, client):
        """No model_key set -> dry run, never silently reports a fake pass/fail."""
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        body = client.get(f"/api/redteam/{sid}").json()
        assert all(r["response"].startswith("[no model]") for r in body["results"])

    def test_run_actually_calls_the_configured_model(self, client_with_model):
        """Regression: model_key was previously collected nowhere and run()
        always used the noop placeholder — every attack trivially 'passed'
        regardless of the target model. Assert the fake adapter's real
        response shows up instead."""
        payload = _session_payload()
        payload["model_key"] = "test-model"
        sid = client_with_model.post("/api/redteam", json=payload).json()["session_id"]
        client_with_model.post(f"/api/redteam/{sid}/run")
        body = client_with_model.get(f"/api/redteam/{sid}").json()
        assert all(r["response"] == "I cannot help with that." for r in body["results"])
        assert all(not r["response"].startswith("[no model]") for r in body["results"])

    def test_run_with_unknown_model_key_becomes_error_not_a_fake_pass_rate(self, client_with_model):
        payload = _session_payload()
        payload["model_key"] = "does-not-exist"
        sid = client_with_model.post("/api/redteam", json=payload).json()["session_id"]
        r = client_with_model.post(f"/api/redteam/{sid}/run")
        assert r.json()["status"] == "error"
        body = client_with_model.get(f"/api/redteam/{sid}").json()
        assert "does-not-exist" in body["error"]
        assert body["results"] == []


class TestResultsEndpoint:
    def test_results_before_run_409(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        r = client.get(f"/api/redteam/{sid}/results")
        assert r.status_code == 409

    def test_results_after_run(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        r = client.get(f"/api/redteam/{sid}/results")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert len(body["results"]) > 0

    def test_results_missing_404(self, client):
        r = client.get("/api/redteam/nonexistent/results")
        assert r.status_code == 404

    def test_result_schema(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        body = client.get(f"/api/redteam/{sid}/results").json()
        r = body["results"][0]
        for key in ("attack_id", "category", "name", "payload", "response", "passed", "reason", "latency_ms"):
            assert key in r

    def test_pass_fail_counts(self, client):
        sid = client.post("/api/redteam", json=_session_payload()).json()["session_id"]
        client.post(f"/api/redteam/{sid}/run")
        body = client.get(f"/api/redteam/{sid}/results").json()
        assert body["passed"] + body["failed"] == len(body["results"])
