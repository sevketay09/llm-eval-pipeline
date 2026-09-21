"""RAG component-level eval API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.config import get_settings
from api.schemas.rag_eval import RagEvalRequest, RagEvalResponse
from api.services.decision_support import require_decision_model
from api.services.rag_eval_service import RagEvalService

router = APIRouter(prefix="/rag-eval", tags=["rag-eval"])

_service = RagEvalService()


@router.post("", response_model=RagEvalResponse)
def evaluate_rag(req: RagEvalRequest):
    if req.decision_model:
        try:
            require_decision_model(req.decision_model, get_settings().models_config_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        return _service.evaluate(
            question=req.question,
            contexts=req.contexts,
            answer=req.answer,
            expected_answer=req.expected_answer,
            embedding_model=req.embedding_model,
            decision_model=req.decision_model,
        )
    except ValueError as exc:
        # Unknown embedding_model key.
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        # Embedding adapter call failed (network/provider error) — the
        # underlying UnifiedEmbeddingAdapter.encode() raises rather than
        # returning a graceful error dict, unlike the LLM adapter.
        raise HTTPException(502, f"Embedding scoring failed: {exc}") from exc
