"""Custom metric API."""
from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException

from api.schemas.custom_metrics import (
    CreateMetricRequest,
    EvaluateMetricRequest,
    EvaluateMetricResponse,
    MetricDetail,
    MetricSummary,
)
from api.services.custom_metric_service import CustomMetricService

router = APIRouter(prefix="/custom-metrics", tags=["custom-metrics"])

_service = CustomMetricService()


def get_service() -> CustomMetricService:
    return _service


@router.post("", response_model=MetricDetail, status_code=201)
def create_metric(
    req: CreateMetricRequest,
    svc: Annotated[CustomMetricService, Depends(get_service)],
):
    rec = svc.create(name=req.name, description=req.description)
    return svc.to_detail(rec)


@router.get("", response_model=List[MetricSummary])
def list_metrics(svc: Annotated[CustomMetricService, Depends(get_service)]):
    return [svc.to_summary(r) for r in svc.list()]


@router.get("/{metric_id}", response_model=MetricDetail)
def get_metric(
    metric_id: str,
    svc: Annotated[CustomMetricService, Depends(get_service)],
):
    rec = svc.get(metric_id)
    if rec is None:
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    return svc.to_detail(rec)


@router.post("/{metric_id}/evaluate", response_model=EvaluateMetricResponse)
def evaluate_metric(
    metric_id: str,
    req: EvaluateMetricRequest,
    svc: Annotated[CustomMetricService, Depends(get_service)],
):
    rec = svc.get(metric_id)
    if rec is None:
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    try:
        return svc.evaluate(metric_id=metric_id, cases=req.cases, judge_model=req.judge_model)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Judge model call failed: {exc}") from exc
