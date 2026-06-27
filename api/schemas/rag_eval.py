"""Pydantic schemas for RAG eval endpoints."""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RagContext(BaseModel):
    text: str
    source: str = ""


class RagEvalRequest(BaseModel):
    question: str = Field(..., min_length=1)
    contexts: List[RagContext] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    expected_answer: str = ""


class RagEvalResponse(BaseModel):
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    fault_component: str
    overall_score: float
    details: Dict[str, Any] = {}
