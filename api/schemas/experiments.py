"""Pydantic schemas for prompt experiment endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptVariantSchema(BaseModel):
    label: str
    system_prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentCaseSchema(BaseModel):
    case_id: str
    input: str
    expected: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateExperimentRequest(BaseModel):
    name: str
    variants: List[PromptVariantSchema] = Field(..., min_length=2, max_length=8)
    dataset: List[ExperimentCaseSchema] = Field(..., min_length=1, max_length=500)
    model_key: str = ""


class RunExperimentRequest(BaseModel):
    base_variant: Optional[str] = None   # label of the "base" for diff; defaults to first
    compare_variant: Optional[str] = None  # label of "compare"; defaults to second


class VariantResultSchema(BaseModel):
    variant_label: str
    case_id: str
    output: str
    score: float
    latency_ms: float
    error: str = ""


class CaseDiffSchema(BaseModel):
    case_id: str
    base_label: str
    compare_label: str
    base_score: float
    compare_score: float
    base_output: str
    compare_output: str
    delta: float
    verdict: str  # improved | regressed | stable | missing


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    model_key: str
    status: str
    variant_count: int
    case_count: int
    created_at: float
    finished_at: Optional[float] = None


class ExperimentDetail(BaseModel):
    experiment_id: str
    name: str
    model_key: str
    variants: List[PromptVariantSchema]
    dataset: List[ExperimentCaseSchema]
    results: List[VariantResultSchema]
    status: str
    error: str = ""
    created_at: float
    finished_at: Optional[float] = None


class CompareResponse(BaseModel):
    experiment_id: str
    base_label: str
    compare_label: str
    diffs: List[CaseDiffSchema]
    improved: int
    regressed: int
    stable: int
    missing: int
    avg_delta: float
