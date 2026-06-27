"""RAG component-level eval API."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas.rag_eval import RagEvalRequest, RagEvalResponse
from api.services.rag_eval_service import RagEvalService

router = APIRouter(prefix="/rag-eval", tags=["rag-eval"])

_service = RagEvalService()


@router.post("", response_model=RagEvalResponse)
def evaluate_rag(req: RagEvalRequest):
    return _service.evaluate(
        question=req.question,
        contexts=req.contexts,
        answer=req.answer,
        expected_answer=req.expected_answer,
    )
