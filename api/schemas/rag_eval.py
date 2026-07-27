"""Pydantic schemas for RAG eval endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RagContext(BaseModel):
    text: str
    source: str = ""


class RagEvalRequest(BaseModel):
    question: str = Field(..., min_length=1)
    contexts: List[RagContext] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    expected_answer: str = ""
    # Optional: score via cosine similarity on this embedding model's vectors
    # instead of the default token-overlap heuristic. Must be a key under
    # config/models.yaml's embedding_models section. Unset -> unchanged
    # (token-overlap) behavior.
    embedding_model: Optional[str] = None


class RagEvalResponse(BaseModel):
    question: str
    context_precision: float
    context_recall: float
    # False when no expected_answer was given — context_recall is then a
    # meaningless 0.0 placeholder (not a real "this failed" score), since
    # analysis.rag_eval.compute_context_recall has nothing to compare
    # against. Additive field so existing consumers reading context_recall
    # as a plain float are unaffected.
    context_recall_applicable: bool = True
    faithfulness: float
    answer_relevance: float
    fault_component: str
    overall_score: float
    scoring_mode: str = "token_overlap"
    embedding_model: Optional[str] = None
    details: Dict[str, Any] = {}


class RagModelAggregate(BaseModel):
    avg_context_precision: float
    # None when no case in this model's results had an expected_answer to
    # compare context against — same "not applicable, not a real zero"
    # distinction as RagEvalResponse.context_recall_applicable, but here
    # it's cheaper to just let the aggregate itself be optional since there's
    # no pre-existing float-typed field to keep non-breaking.
    avg_context_recall: Optional[float] = None
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_overall_rag_score: float
    fault_distribution: Dict[str, int]
    rag_case_count: int


class RagReportEvalResponse(BaseModel):
    total_rag_cases: int
    models: Dict[str, RagModelAggregate]
    overall_fault_distribution: Dict[str, int]
    scoring_mode: str = "token_overlap"
    embedding_model: Optional[str] = None
