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


@pytest.fixture()
def client():
    app = FastAPI()
    svc = RedTeamService(store=RedTeamStore())
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
