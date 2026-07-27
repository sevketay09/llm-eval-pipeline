"""Prompt experiment API."""
from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.experiments import (
    CompareResponse,
    CreateExperimentRequest,
    ExperimentDetail,
    ExperimentSummary,
    RunExperimentRequest,
)
from api.services.experiment_service import ExperimentService
from experiments.store import ExperimentCase, PromptVariant

router = APIRouter(prefix="/experiments", tags=["experiments"])

_service = ExperimentService()


def get_service() -> ExperimentService:
    return _service


@router.post("", response_model=ExperimentSummary, status_code=201)
def create_experiment(
    req: CreateExperimentRequest,
    svc: Annotated[ExperimentService, Depends(get_service)],
):
    variants = [PromptVariant(label=v.label, system_prompt=v.system_prompt, metadata=v.metadata) for v in req.variants]
    dataset = [ExperimentCase(case_id=c.case_id, input=c.input, expected=c.expected, metadata=c.metadata) for c in req.dataset]
    exp = svc.create(name=req.name, variants=variants, dataset=dataset, model_key=req.model_key)
    return svc.to_summary(exp)


@router.get("", response_model=List[ExperimentSummary])
def list_experiments(
    svc: Annotated[ExperimentService, Depends(get_service)],
    limit: int = Query(50, ge=1, le=200),
):
    return [svc.to_summary(e) for e in svc.list(limit=limit)]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(
    experiment_id: str,
    svc: Annotated[ExperimentService, Depends(get_service)],
):
    exp = svc.get(experiment_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    return svc.to_detail(exp)


@router.post("/{experiment_id}/run", status_code=202, response_model=ExperimentSummary)
async def run_experiment(
    experiment_id: str,
    req: RunExperimentRequest,
    svc: Annotated[ExperimentService, Depends(get_service)],
):
    exp = svc.get(experiment_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    if exp.status == "running":
        raise HTTPException(409, "Experiment is already running")
    exp = await svc.run(experiment_id)
    return svc.to_summary(exp)


@router.get("/{experiment_id}/compare", response_model=CompareResponse)
def compare_experiment(
    experiment_id: str,
    svc: Annotated[ExperimentService, Depends(get_service)],
    base: Optional[str] = Query(None, description="Base variant label"),
    compare: Optional[str] = Query(None, description="Compare variant label"),
):
    exp = svc.get(experiment_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    if exp.status != "done":
        raise HTTPException(409, f"Experiment is not done yet (status={exp.status})")
    result = svc.compare(experiment_id, base_variant=base, compare_variant=compare)
    if result is None:
        raise HTTPException(422, "No results to compare")
    return result
