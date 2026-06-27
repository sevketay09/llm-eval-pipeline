"""Pydantic schemas for trace ingest/query endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SpanSchema(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    type: str = "GENERIC"
    input: Any = None
    output: Any = None
    latency_ms: float = 0.0
    start_ts: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceSchema(BaseModel):
    trace_id: str
    name: str
    tags: List[str] = Field(default_factory=list)
    spans: List[SpanSchema] = Field(default_factory=list)
    start_ts: float = 0.0
    end_ts: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# POST /api/traces/ingest accepts single or batch
TraceIngestRequest = Union[TraceSchema, List[TraceSchema]]


class TraceListResponse(BaseModel):
    traces: List[TraceSchema]
    total: int


class TraceDetail(BaseModel):
    trace: TraceSchema
    span_count: int
    duration_ms: Optional[float] = None
