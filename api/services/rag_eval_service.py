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

        # evaluate_rag_case() returns metrics at the top level (context_precision,
        # context_recall, faithfulness, answer_relevance, fault_isolation), not
        # nested under a "scores" key — reading via result.get("scores", {})
        # always got an empty dict, so every field below silently fell back to
        # its 0.0/"none" default regardless of the real computed values (which
        # were still correct, just discarded). context_recall's "recall" is
        # None (not 0.0) when no expected_answer was given — `or 0.0` handles
        # both that and a genuine 0.0 score identically.
        cp = result.get("context_precision", {}).get("precision") or 0.0
        cr = result.get("context_recall", {}).get("recall") or 0.0
        fa = result.get("faithfulness", {}).get("faithfulness") or 0.0
        ar = result.get("answer_relevance", {}).get("answer_relevance") or 0.0
        fault = result.get("fault_isolation", {}).get("fault", "none")
        overall = round(result.get("overall_rag_score", 0.0), 4)

        return RagEvalResponse(
            question=question,
            context_precision=round(cp, 4),
            context_recall=round(cr, 4),
            faithfulness=round(fa, 4),
            answer_relevance=round(ar, 4),
            fault_component=fault,
            overall_score=overall,
            details=result,
        )
