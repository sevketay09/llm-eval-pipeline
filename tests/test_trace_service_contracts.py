"""Contract tests for api/services/trace_service.py — no HTTP."""
from __future__ import annotations

import asyncio
import uuid
import pytest

from api.schemas.traces import TraceSchema, SpanSchema
from api.services.trace_service import TraceStore, _MAX_TRACES


def make_trace(trace_id: str = None, tags=None, run_id: str = None) -> TraceSchema:
    return TraceSchema(
        trace_id=trace_id or uuid.uuid4().hex,
        name="test_fn",
        tags=tags or [],
        metadata={"run_id": run_id} if run_id else {},
    )


class TestTraceStoreIngest:
    def test_ingest_single_returns_id(self):
        store = TraceStore()
        t = make_trace("t1")
        ids = asyncio.get_event_loop().run_until_complete(store.ingest([t]))
        assert ids == ["t1"]

    def test_ingest_batch(self):
        store = TraceStore()
        traces = [make_trace(f"t{i}") for i in range(5)]
        ids = asyncio.get_event_loop().run_until_complete(store.ingest(traces))
        assert len(ids) == 5
        assert store.count() == 5

    def test_ingest_duplicate_overwrites(self):
        store = TraceStore()
        t1 = make_trace("dup")
        t2 = TraceSchema(trace_id="dup", name="updated_fn")
        asyncio.get_event_loop().run_until_complete(store.ingest([t1]))
        asyncio.get_event_loop().run_until_complete(store.ingest([t2]))
        assert store.count() == 1

    def test_ingest_updates_duplicate_data(self):
        store = TraceStore()
        t1 = make_trace("dup")
        t2 = TraceSchema(trace_id="dup", name="updated_fn")
        asyncio.get_event_loop().run_until_complete(store.ingest([t1]))
        asyncio.get_event_loop().run_until_complete(store.ingest([t2]))
        result = asyncio.get_event_loop().run_until_complete(store.get("dup"))
        assert result.name == "updated_fn"


class TestTraceStoreGet:
    def test_get_existing(self):
        store = TraceStore()
        t = make_trace("tid1")
        asyncio.get_event_loop().run_until_complete(store.ingest([t]))
        result = asyncio.get_event_loop().run_until_complete(store.get("tid1"))
        assert result is not None
        assert result.trace_id == "tid1"

    def test_get_missing_returns_none(self):
        store = TraceStore()
        result = asyncio.get_event_loop().run_until_complete(store.get("nonexistent"))
        assert result is None


class TestTraceStoreList:
    def test_list_all(self):
        store = TraceStore()
        traces = [make_trace() for _ in range(3)]
        asyncio.get_event_loop().run_until_complete(store.ingest(traces))
        results = asyncio.get_event_loop().run_until_complete(store.list())
        assert len(results) == 3

    def test_list_filter_tag(self):
        store = TraceStore()
        asyncio.get_event_loop().run_until_complete(store.ingest([
            make_trace(tags=["prod"]),
            make_trace(tags=["dev"]),
            make_trace(tags=["prod"]),
        ]))
        results = asyncio.get_event_loop().run_until_complete(store.list(tag="prod"))
        assert len(results) == 2

    def test_list_filter_run_id(self):
        store = TraceStore()
        asyncio.get_event_loop().run_until_complete(store.ingest([
            make_trace(run_id="run-1"),
            make_trace(run_id="run-2"),
            make_trace(run_id="run-1"),
        ]))
        results = asyncio.get_event_loop().run_until_complete(store.list(run_id="run-1"))
        assert len(results) == 2

    def test_list_limit(self):
        store = TraceStore()
        traces = [make_trace() for _ in range(20)]
        asyncio.get_event_loop().run_until_complete(store.ingest(traces))
        results = asyncio.get_event_loop().run_until_complete(store.list(limit=5))
        assert len(results) == 5


class TestTraceStoreFifoEviction:
    def test_evicts_oldest_at_max(self):
        store = TraceStore()
        # Ingest MAX+10 traces
        first_id = "first-trace"
        first = [make_trace(first_id)]
        rest = [make_trace() for _ in range(_MAX_TRACES + 9)]
        asyncio.get_event_loop().run_until_complete(store.ingest(first))
        asyncio.get_event_loop().run_until_complete(store.ingest(rest))
        assert store.count() <= _MAX_TRACES
        # first_id should be evicted
        result = asyncio.get_event_loop().run_until_complete(store.get(first_id))
        assert result is None


class TestTraceStoreDelete:
    def test_delete_existing(self):
        store = TraceStore()
        t = make_trace("del1")
        asyncio.get_event_loop().run_until_complete(store.ingest([t]))
        ok = asyncio.get_event_loop().run_until_complete(store.delete("del1"))
        assert ok is True
        assert store.count() == 0

    def test_delete_missing_returns_false(self):
        store = TraceStore()
        ok = asyncio.get_event_loop().run_until_complete(store.delete("nope"))
        assert ok is False


class TestTraceStorePersistence:
    def test_save_and_load_from_round_trips(self, tmp_path):
        path = tmp_path / "traces.json"
        loop = asyncio.get_event_loop()
        store = TraceStore()
        t = make_trace("t1", tags=["eval_sampled"], run_id="run-1")
        loop.run_until_complete(store.ingest([t]))
        loop.run_until_complete(store.save(path))

        reloaded = TraceStore()
        loop.run_until_complete(reloaded.load_from(path))

        restored = loop.run_until_complete(reloaded.get("t1"))
        assert restored is not None
        assert restored.name == "test_fn"
        assert "eval_sampled" in restored.tags

    def test_load_from_missing_file_is_noop(self, tmp_path):
        store = TraceStore()
        asyncio.get_event_loop().run_until_complete(store.load_from(tmp_path / "does-not-exist.json"))
        assert store.count() == 0

    def test_save_without_path_is_noop(self):
        store = TraceStore()
        # No dump_path configured and no explicit path passed — should not raise.
        asyncio.get_event_loop().run_until_complete(store.save())
