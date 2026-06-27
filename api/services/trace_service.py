"""In-memory trace store with asyncio lock and FIFO eviction."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

from api.schemas.traces import TraceSchema

_MAX_TRACES = 10_000


class TraceStore:
    def __init__(self, dump_path: Optional[Path] = None):
        self._store: Dict[str, TraceSchema] = {}
        self._order: List[str] = []
        self._lock = asyncio.Lock()
        self._dump_path = dump_path

    async def ingest(self, traces: List[TraceSchema]) -> List[str]:
        async with self._lock:
            ids: List[str] = []
            for t in traces:
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

    async def delete(self, trace_id: str) -> bool:
        async with self._lock:
            if trace_id not in self._store:
                return False
            del self._store[trace_id]
            if trace_id in self._order:
                self._order.remove(trace_id)
            return True

    async def dump(self) -> None:
        if not self._dump_path:
            return
        async with self._lock:
            data = [t.model_dump() for t in self._store.values()]
        self._dump_path.write_text(json.dumps(data, indent=2))

    def count(self) -> int:
        return len(self._store)
