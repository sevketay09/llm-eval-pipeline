"""Contract tests for api/routers/traces.py — TestClient, no network."""
from __future__ import annotations

import uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.traces import router as traces_router, get_store
from api.services.trace_service import TraceStore


@pytest.fixture()
def client():
    """Minimal app with only traces router — avoids pulling scipy/ml deps."""
    app = FastAPI()
    store = TraceStore()
    app.dependency_overrides[get_store] = lambda: store
    app.include_router(traces_router, prefix="/api")
    with TestClient(app) as c:
        yield c


def _trace_payload(trace_id: str = None, tags=None) -> dict:
    return {
        "trace_id": trace_id or uuid.uuid4().hex,
        "name": "test_fn",
        "tags": tags or [],
        "spans": [],
        "start_ts": 1.0,
    }


class TestIngestEndpoint:
    def test_ingest_single_202(self, client):
        r = client.post("/api/traces/ingest", json=_trace_payload("t1"))
        assert r.status_code == 202
        assert r.json()["ingested"] == 1

    def test_ingest_batch_202(self, client):
        payload = [_trace_payload(f"t{i}") for i in range(3)]
        r = client.post("/api/traces/ingest", json=payload)
        assert r.status_code == 202
        assert r.json()["ingested"] == 3

    def test_ingest_returns_trace_ids(self, client):
        r = client.post("/api/traces/ingest", json=_trace_payload("myid"))
        assert "myid" in r.json()["trace_ids"]


class TestListEndpoint:
    def test_list_empty(self, client):
        r = client.get("/api/traces")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_list_after_ingest(self, client):
        for i in range(3):
            client.post("/api/traces/ingest", json=_trace_payload(f"lt{i}"))
        r = client.get("/api/traces")
        assert r.json()["total"] == 3

    def test_list_filter_tag(self, client):
        client.post("/api/traces/ingest", json={**_trace_payload("a"), "tags": ["prod"]})
        client.post("/api/traces/ingest", json={**_trace_payload("b"), "tags": ["dev"]})
        r = client.get("/api/traces?tag=prod")
        assert r.json()["total"] == 1

    def test_list_limit(self, client):
        for i in range(10):
            client.post("/api/traces/ingest", json=_trace_payload(f"lim{i}"))
        r = client.get("/api/traces?limit=3")
        assert len(r.json()["traces"]) == 3


class TestDetailEndpoint:
    def test_get_existing_trace(self, client):
        client.post("/api/traces/ingest", json=_trace_payload("detail1"))
        r = client.get("/api/traces/detail1")
        assert r.status_code == 200
        assert r.json()["trace"]["trace_id"] == "detail1"

    def test_get_missing_404(self, client):
        r = client.get("/api/traces/doesnotexist")
        assert r.status_code == 404

    def test_span_count_zero(self, client):
        client.post("/api/traces/ingest", json=_trace_payload("sc1"))
        r = client.get("/api/traces/sc1")
        assert r.json()["span_count"] == 0

    def test_span_count_with_spans(self, client):
        payload = {
            **_trace_payload("sc2"),
            "spans": [
                {"span_id": "s1", "name": "step", "type": "LLM", "latency_ms": 10.0},
                {"span_id": "s2", "name": "step2", "type": "TOOL", "latency_ms": 5.0},
            ],
        }
        client.post("/api/traces/ingest", json=payload)
        r = client.get("/api/traces/sc2")
        assert r.json()["span_count"] == 2


class TestEvalEndpoint:
    def test_eval_existing_202(self, client):
        client.post("/api/traces/ingest", json=_trace_payload("ev1"))
        r = client.post("/api/traces/ev1/eval")
        assert r.status_code == 202
        assert r.json()["status"] == "eval_queued"

    def test_eval_missing_404(self, client):
        r = client.post("/api/traces/missing/eval")
        assert r.status_code == 404

    def test_eval_returns_trace_id(self, client):
        client.post("/api/traces/ingest", json=_trace_payload("ev2"))
        r = client.post("/api/traces/ev2/eval")
        assert r.json()["trace_id"] == "ev2"
