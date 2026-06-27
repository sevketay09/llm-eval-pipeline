"""Contract tests: OnlineSampler wiring into TraceStore and eval_trace endpoint."""
from __future__ import annotations
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.traces import router as traces_router, get_store
from api.services.trace_service import TraceStore, _SAMPLED_TAG
from tracing.sampler import OnlineSampler


@pytest.fixture(autouse=True)
def fresh_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _schema(trace_id: str, tags: list | None = None):
    from api.schemas.traces import TraceSchema
    return TraceSchema(trace_id=trace_id, name="test", tags=tags or [])


# ── TraceStore + OnlineSampler unit contracts ─────────────────────────────────

class TestTraceStoreAutoTag:
    def test_auto_tag_when_sampled(self):
        store = TraceStore(sampler=OnlineSampler(rate=1.0))
        _run(store.ingest([_schema("t1")]))
        t = _run(store.get("t1"))
        assert _SAMPLED_TAG in t.tags

    def test_no_tag_when_not_sampled(self):
        store = TraceStore(sampler=OnlineSampler(rate=0.0))
        _run(store.ingest([_schema("t1")]))
        t = _run(store.get("t1"))
        assert _SAMPLED_TAG not in t.tags

    def test_no_duplicate_tag_on_reingest(self):
        store = TraceStore(sampler=OnlineSampler(rate=1.0))
        _run(store.ingest([_schema("t1")]))
        _run(store.ingest([_schema("t1")]))
        t = _run(store.get("t1"))
        assert t.tags.count(_SAMPLED_TAG) == 1

    def test_no_sampler_leaves_tags_unchanged(self):
        store = TraceStore()
        _run(store.ingest([_schema("t1", tags=["prod"])]))
        t = _run(store.get("t1"))
        assert t.tags == ["prod"]

    def test_tag_method_adds_tag(self):
        store = TraceStore()
        _run(store.ingest([_schema("t1")]))
        result = _run(store.tag("t1", _SAMPLED_TAG))
        assert result is True
        t = _run(store.get("t1"))
        assert _SAMPLED_TAG in t.tags

    def test_tag_method_returns_false_for_missing(self):
        store = TraceStore()
        result = _run(store.tag("nonexistent", _SAMPLED_TAG))
        assert result is False

    def test_tag_idempotent(self):
        store = TraceStore()
        _run(store.ingest([_schema("t1")]))
        _run(store.tag("t1", _SAMPLED_TAG))
        _run(store.tag("t1", _SAMPLED_TAG))
        t = _run(store.get("t1"))
        assert t.tags.count(_SAMPLED_TAG) == 1

    def test_list_by_sampled_tag(self):
        store = TraceStore(sampler=OnlineSampler(rate=1.0))
        _run(store.ingest([_schema("t1"), _schema("t2")]))
        results = _run(store.list(tag=_SAMPLED_TAG))
        assert {t.trace_id for t in results} == {"t1", "t2"}


# ── eval_trace endpoint contracts ─────────────────────────────────────────────

def _make_client(rate: float):
    app = FastAPI()
    store = TraceStore(sampler=OnlineSampler(rate=rate))
    app.dependency_overrides[get_store] = lambda: store
    app.include_router(traces_router)
    return TestClient(app), store


class TestEvalTraceEndpoint:
    def test_eval_404_for_missing(self):
        client, _ = _make_client(1.0)
        r = client.post("/traces/nope/eval")
        assert r.status_code == 404

    def test_eval_queued_when_sampled(self):
        client, _ = _make_client(1.0)
        client.post("/traces/ingest", json={"trace_id": "t1", "name": "n"})
        r = client.post("/traces/t1/eval")
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        assert body["sampled"] is True

    def test_eval_skipped_when_not_sampled(self):
        client, _ = _make_client(0.0)
        client.post("/traces/ingest", json={"trace_id": "t1", "name": "n"})
        r = client.post("/traces/t1/eval")
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "skipped"
        assert body["sampled"] is False

    def test_eval_tags_trace_in_store(self):
        client, _ = _make_client(1.0)
        client.post("/traces/ingest", json={"trace_id": "t1", "name": "n"})
        client.post("/traces/t1/eval")
        r = client.get("/traces/t1")
        assert _SAMPLED_TAG in r.json()["trace"]["tags"]

    def test_eval_already_sampled_returns_queued(self):
        """Trace already tagged eval_sampled → queued regardless of sampler rate."""
        client, _ = _make_client(0.0)
        client.post("/traces/ingest", json={"trace_id": "t1", "name": "n", "tags": [_SAMPLED_TAG]})
        r = client.post("/traces/t1/eval")
        body = r.json()
        assert body["status"] == "queued"
        assert body["sampled"] is True

    def test_eval_returns_span_count(self):
        client, _ = _make_client(1.0)
        payload = {
            "trace_id": "t1",
            "name": "n",
            "spans": [
                {"span_id": "s1", "name": "llm", "type": "LLM", "latency_ms": 50.0, "start_ts": 1.0},
                {"span_id": "s2", "name": "ret", "type": "RETRIEVER", "latency_ms": 20.0, "start_ts": 1.0},
            ],
        }
        client.post("/traces/ingest", json=payload)
        r = client.post("/traces/t1/eval")
        assert r.json()["span_count"] == 2

    def test_ingest_auto_tags_sampled(self):
        client, _ = _make_client(1.0)
        client.post("/traces/ingest", json={"trace_id": "t1", "name": "n"})
        r = client.get("/traces/t1")
        assert _SAMPLED_TAG in r.json()["trace"]["tags"]
