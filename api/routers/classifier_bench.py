"""Classifier & Guardrail Bench API."""
from __future__ import annotations

import asyncio
import json
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException

from api.config import get_settings
from api.rate_limit import RateLimiter
from api.schemas.classifier_bench import (
    BenchCase,
    BenchDetail,
    BenchSummary,
    CreateBenchRequest,
    ImportJsonlRequest,
    ImportJsonlResponse,
)
from api.services.classifier_bench_service import ClassifierBenchService
from api.services.decision_support import ensure_not_decision_model, require_decision_model

router = APIRouter(prefix="/classifier-bench", tags=["classifier-bench"])

_service = ClassifierBenchService()
_run_rate_limit = RateLimiter("bench_run", limit=5, window_seconds=60)


def get_service() -> ClassifierBenchService:
    return _service


@router.post("", response_model=BenchSummary, status_code=201)
def create_bench(
    req: CreateBenchRequest,
    svc: Annotated[ClassifierBenchService, Depends(get_service)],
):
    config_path = get_settings().models_config_path
    try:
        require_decision_model(req.decision_model, config_path)
        ensure_not_decision_model(req.llm_models, config_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    bench = svc.create(req)
    return svc.to_summary(bench)


@router.get("", response_model=List[BenchSummary])
def list_benches(svc: Annotated[ClassifierBenchService, Depends(get_service)]):
    return [svc.to_summary(b) for b in svc.list()]


@router.get("/{bench_id}", response_model=BenchDetail)
def get_bench(
    bench_id: str,
    svc: Annotated[ClassifierBenchService, Depends(get_service)],
):
    bench = svc.get(bench_id)
    if bench is None:
        raise HTTPException(404, f"Bench '{bench_id}' not found")
    return svc.to_detail(bench)


@router.post(
    "/{bench_id}/run",
    response_model=BenchSummary,
    status_code=202,
    dependencies=[Depends(_run_rate_limit)],
)
async def run_bench(
    bench_id: str,
    svc: Annotated[ClassifierBenchService, Depends(get_service)],
):
    bench = svc.get(bench_id)
    if bench is None:
        raise HTTPException(404, f"Bench '{bench_id}' not found")
    if bench["status"] == "running":
        raise HTTPException(409, "Bench is already running")
    bench = await asyncio.get_running_loop().run_in_executor(None, svc.run, bench_id)
    return svc.to_summary(bench)


@router.post("/import-jsonl", response_model=ImportJsonlResponse)
def import_jsonl(req: ImportJsonlRequest):
    cases: List[BenchCase] = []
    for line_no, line in enumerate(req.jsonl_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Line {line_no} is not valid JSON: {exc}") from exc
        cases.append(
            BenchCase(
                id=str(raw.get("id") or f"case_{line_no:04d}"),
                text=str(raw.get("message") or raw.get("text") or ""),
                label=raw.get("true_label") or raw.get("label"),
                expected_verdict=raw.get("expected_verdict"),
                labels=raw.get("labels") or [],
            )
        )
    if not cases:
        raise HTTPException(400, "No cases found in JSONL input")
    return ImportJsonlResponse(cases=cases)
