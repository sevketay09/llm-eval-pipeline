"""In-memory trace store with asyncio lock and FIFO eviction."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from api.schemas.traces import TraceSchema

if TYPE_CHECKING:
    from tracing.sampler import OnlineSampler

_MAX_TRACES = 10_000
_SAMPLED_TAG = "eval_sampled"


class TraceStore:
    def __init__(
        self,
        dump_path: Optional[Path] = None,
        sampler: Optional["OnlineSampler"] = None,
    ):
        self._store: Dict[str, TraceSchema] = {}
        self._order: List[str] = []
        self._lock = asyncio.Lock()
        self._dump_path = dump_path
        self._sampler = sampler

    async def ingest(self, traces: List[TraceSchema]) -> List[str]:
        async with self._lock:
            ids: List[str] = []
            for t in traces:
                if self._sampler and _SAMPLED_TAG not in t.tags and self._sampler.sample(t.trace_id):
                    t = t.model_copy(update={"tags": list(t.tags) + [_SAMPLED_TAG]})
                if t.trace_id not in self._store:
                    self._order.append(t.trace_id)
                self._store[t.trace_id] = t
                ids.append(t.trace_id)
            # FIFO evict oldest when over limit
            while len(self._order) > _MAX_TRACES:
                evicted = self._order.pop(0)
                self._store.pop(evicted, None)
            return ids

    async def get(self, trace_id: str) -> Optional[TraceSchema]:
        async with self._lock:
            return self._store.get(trace_id)

    async def list(
        self,
        run_id: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
    ) -> List[TraceSchema]:
        async with self._lock:
            results = list(self._store.values())
        if run_id is not None:
            results = [t for t in results if t.metadata.get("run_id") == run_id]
        if tag is not None:
            results = [t for t in results if tag in t.tags]
        return results[-limit:]

    async def tag(self, trace_id: str, tag_value: str) -> bool:
        """Add a tag to an existing trace. Returns False if trace not found."""
        async with self._lock:
            t = self._store.get(trace_id)
            if t is None:
                return False
            if tag_value not in t.tags:
                self._store[trace_id] = t.model_copy(update={"tags": list(t.tags) + [tag_value]})
            return True

    async def delete(self, trace_id: str) -> bool:
        async with self._lock:
            if trace_id not in self._store:
                return False
            del self._store[trace_id]
            if trace_id in self._order:
                self._order.remove(trace_id)
            return True

    async def save(self, path: Optional[Path] = None) -> None:
        """Snapshot current state to disk (atomic write) so a process restart doesn't lose it."""
        target = path or self._dump_path
        if not target:
            return
        async with self._lock:
            data = [self._store[tid].model_dump(mode="json") for tid in self._order if tid in self._store]
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(target)

    async def load_from(self, path: Optional[Path] = None) -> None:
        """Replace current state with a previously saved snapshot, if one exists."""
        target = path or self._dump_path
        if not target or not target.exists():
            return
        raw = json.loads(target.read_text())
        async with self._lock:
            self._store = {}
            self._order = []
            for item in raw:
                t = TraceSchema.model_validate(item)
                self._order.append(t.trace_id)
                self._store[t.trace_id] = t

    def count(self) -> int:
        return len(self._store)
