"""Failure clustering API."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas.failure_clustering import (
    FailureClusteringRequest,
    FailureClusteringResponse,
)
from api.services.failure_clustering_service import FailureClusteringService

router = APIRouter(prefix="/failure-clustering", tags=["failure-clustering"])

_service = FailureClusteringService()


@router.post("", response_model=FailureClusteringResponse)
def cluster_failures(req: FailureClusteringRequest):
    return _service.cluster(
        report=req.report,
        threshold=req.threshold,
        n_clusters=req.n_clusters,
    )
