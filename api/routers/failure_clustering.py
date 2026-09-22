"""Failure clustering API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.config import get_settings
from api.schemas.failure_clustering import (
    FailureClusteringRequest,
    FailureClusteringResponse,
)
from api.services.decision_support import require_decision_model
from api.services.failure_clustering_service import FailureClusteringService

router = APIRouter(prefix="/failure-clustering", tags=["failure-clustering"])

_service = FailureClusteringService()


@router.post("", response_model=FailureClusteringResponse)
def cluster_failures(req: FailureClusteringRequest):
    if req.labeling == "decision":
        if not req.decision_model:
            raise HTTPException(400, "decision_model is required when labeling='decision'")
        try:
            require_decision_model(req.decision_model, get_settings().models_config_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return _service.cluster(
        report=req.report,
        threshold=req.threshold,
        n_clusters=req.n_clusters,
        labeling=req.labeling,
        decision_model=req.decision_model,
        taxonomy=req.taxonomy,
    )
