"""Trace ingest and query API."""
from __future__ import annotations

from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.traces import TraceDetail, TraceIngestRequest, TraceListResponse, TraceSchema
from api.services.trace_service import TraceStore

router = APIRouter(prefix="/traces", tags=["traces"])

_store = TraceStore()


def get_store() -> TraceStore:
    return _store


@router.post("/ingest", status_code=202)
async def ingest_traces(
    body: Union[TraceSchema, List[TraceSchema]],
    store: Annotated[TraceStore, Depends(get_store)],
):
    traces = body if isinstance(body, list) else [body]
    ids = await store.ingest(traces)
    return {"ingested": len(ids), "trace_ids": ids}


@router.get("", response_model=TraceListResponse)
async def list_traces(
    store: Annotated[TraceStore, Depends(get_store)],
    run_id: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    traces = await store.list(run_id=run_id, tag=tag, limit=limit)
    return TraceListResponse(traces=traces, total=len(traces))


@router.get("/{trace_id}", response_model=TraceDetail)
async def get_trace(
    trace_id: str,
    store: Annotated[TraceStore, Depends(get_store)],
):
    t = await store.get(trace_id)
    if t is None:
        raise HTTPException(404, f"Trace '{trace_id}' not found")
    duration: Optional[float] = None
    if t.end_ts is not None and t.start_ts:
        duration = (t.end_ts - t.start_ts) * 1000
    return TraceDetail(trace=t, span_count=len(t.spans), duration_ms=duration)


@router.post("/{trace_id}/eval", status_code=202)
async def eval_trace(
    trace_id: str,
    store: Annotated[TraceStore, Depends(get_store)],
):
    t = await store.get(trace_id)
    if t is None:
        raise HTTPException(404, f"Trace '{trace_id}' not found")
    return {"trace_id": trace_id, "status": "eval_queued", "span_count": len(t.spans)}
