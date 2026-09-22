"""Trace ingest and query API."""
from __future__ import annotations

import asyncio
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query

from api.config import get_settings
from api.schemas.traces import (
    DecideBatchRequest,
    DecideBatchResponse,
    DecideTraceRequest,
    DecideTraceResponse,
    ToHitlResponse,
    TraceDetail,
    TraceIngestRequest,
    TraceListResponse,
    TraceSchema,
)
from api.services.trace_decision_service import TraceDecisionService
from api.services.trace_service import TraceStore, _SAMPLED_TAG
from tracing.sampler import OnlineSampler
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

_sampler = OnlineSampler(rate=0.1)
_store = TraceStore(sampler=_sampler)
_decision_service = TraceDecisionService(_store)


def get_store() -> TraceStore:
    return _store


def get_decision_service() -> TraceDecisionService:
    return _decision_service


async def _auto_decide(trace_ids: List[str], decision_model: str) -> None:
    try:
        await _decision_service.decide_batch(decision_model, trace_ids=trace_ids)
    except Exception as exc:  # noqa: BLE001 — background task must never crash ingest
        logger.warning("[traces] auto decide_batch failed: %s", exc)


@router.post("/ingest", status_code=202)
async def ingest_traces(
    body: Union[TraceSchema, List[TraceSchema]],
    store: Annotated[TraceStore, Depends(get_store)],
):
    traces = body if isinstance(body, list) else [body]
    ids = await store.ingest(traces)
    decision_model = get_settings().trace_decision_model
    if decision_model:
        asyncio.create_task(_auto_decide(ids, decision_model))
    return {"ingested": len(ids), "trace_ids": ids}


@router.post("/decide-batch", response_model=DecideBatchResponse)
async def decide_batch_traces(
    req: DecideBatchRequest,
    svc: Annotated[TraceDecisionService, Depends(get_decision_service)],
):
    result = await svc.decide_batch(
        req.decision_model, trace_ids=req.trace_ids, tag=req.tag, limit=req.limit, force=req.force
    )
    return DecideBatchResponse(**result)


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
    already_sampled = _SAMPLED_TAG in t.tags
    sampled = already_sampled or _sampler.sample(trace_id)
    if sampled and not already_sampled:
        await store.tag(trace_id, _SAMPLED_TAG)
    return {
        "trace_id": trace_id,
        "status": "queued" if sampled else "skipped",
        "sampled": sampled,
        "span_count": len(t.spans),
    }


@router.post("/{trace_id}/decide", response_model=DecideTraceResponse)
async def decide_trace_endpoint(
    trace_id: str,
    req: DecideTraceRequest,
    svc: Annotated[TraceDecisionService, Depends(get_decision_service)],
):
    result = await svc.decide(trace_id, req.decision_model)
    if result is None:
        raise HTTPException(404, f"Trace '{trace_id}' not found")
    return DecideTraceResponse(**result)


@router.post("/{trace_id}/to-hitl", response_model=ToHitlResponse)
async def trace_to_hitl(
    trace_id: str,
    store: Annotated[TraceStore, Depends(get_store)],
    svc: Annotated[TraceDecisionService, Depends(get_decision_service)],
):
    t = await store.get(trace_id)
    if t is None:
        raise HTTPException(404, f"Trace '{trace_id}' not found")
    item_id = await svc.to_hitl(trace_id)
    if item_id is None:
        raise HTTPException(409, f"Trace '{trace_id}' is not flagged for review")
    return ToHitlResponse(item_id=item_id)
