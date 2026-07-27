"""Pydantic schemas for custom metric endpoints."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CreateMetricRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class EvaluateCaseRequest(BaseModel):
    question: str
    answer: str
    expected_answer: str = ""


class EvaluateMetricRequest(BaseModel):
    cases: List[EvaluateCaseRequest] = Field(..., min_length=1)
    judge_model: Optional[str] = None


class CaseEvalResult(BaseModel):
    question: str
    answer: str
    expected_answer: str
    score: Optional[float] = None
    reasoning: str = ""
    error: str = ""


class MetricSummary(BaseModel):
    metric_id: str
    name: str
    description: str
    status: str
    created_at: float


class MetricDetail(MetricSummary):
    prompt: str


class EvaluateMetricResponse(BaseModel):
    metric_id: str
    name: str
    results: List[CaseEvalResult]
    avg_score: Optional[float] = None
