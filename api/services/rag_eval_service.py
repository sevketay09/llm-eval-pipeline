"""RAG eval service — wraps analysis/rag_eval."""
from __future__ import annotations

from typing import List

from api.schemas.rag_eval import RagContext, RagEvalResponse
from analysis.rag_eval import evaluate_rag_case


class RagEvalService:
    def evaluate(
        self,
        question: str,
        contexts: List[RagContext],
        answer: str,
        expected_answer: str = "",
    ) -> RagEvalResponse:
        context_texts = [c.text for c in contexts]
        case = {
            "question": question,
            "contexts": context_texts,
            "answer": answer,
            "expected_answer": expected_answer,
        }
        result = evaluate_rag_case(case)

        scores = result.get("scores", {})
        fault = result.get("fault_isolation", {})

        cp = scores.get("context_precision", {}).get("score", 0.0)
        cr = scores.get("context_recall", {}).get("score", 0.0)
        fa = scores.get("faithfulness", {}).get("score", 0.0)
        ar = scores.get("answer_relevance", {}).get("score", 0.0)
        overall = round((cp + cr + fa + ar) / 4, 4)

        return RagEvalResponse(
            question=question,
            context_precision=round(cp, 4),
            context_recall=round(cr, 4),
            faithfulness=round(fa, 4),
            answer_relevance=round(ar, 4),
            fault_component=fault.get("fault_component", "none"),
            overall_score=overall,
            details=result,
        )
