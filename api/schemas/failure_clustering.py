"""Pydantic schemas for failure clustering endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class FailureClusteringRequest(BaseModel):
    report: Dict[str, Any]
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    n_clusters: Optional[int] = Field(None, ge=2, le=20)
    labeling: Literal["keywords", "decision"] = "keywords"
    decision_model: Optional[str] = Field(None, description="Required when labeling='decision'")
    taxonomy: Optional[Dict[str, str]] = Field(None, description="Custom label -> description; defaults to DEFAULT_FAILURE_TAXONOMY")


class ClusterMemberSchema(BaseModel):
    model: str
    test: str
    case_id: str
    score: float
    category: str
    text: str
    taxonomy_label: Optional[str] = None


class ClusterSchema(BaseModel):
    cluster_id: int
    size: int
    label: str
    centroid_text: str
    avg_score: float
    members: List[ClusterMemberSchema]
    taxonomy_label: Optional[str] = None
    label_distribution: Dict[str, int] = Field(default_factory=dict)


class FailureClusteringResponse(BaseModel):
    total_failures: int
    threshold: float
    clusters: List[ClusterSchema]
    model_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    taxonomy_breakdown: Dict[str, int] = Field(default_factory=dict)
