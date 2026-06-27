"""Pydantic schemas for failure clustering endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureClusteringRequest(BaseModel):
    report: Dict[str, Any]
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    n_clusters: Optional[int] = Field(None, ge=2, le=20)


class ClusterMemberSchema(BaseModel):
    model: str
    test: str
    case_id: str
    score: float
    category: str
    text: str


class ClusterSchema(BaseModel):
    cluster_id: int
    size: int
    label: str
    centroid_text: str
    avg_score: float
    members: List[ClusterMemberSchema]


class FailureClusteringResponse(BaseModel):
    total_failures: int
    threshold: float
    clusters: List[ClusterSchema]
    model_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
