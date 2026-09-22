"""Jev-backed RAG evaluation — one call per case instead of four lexical
metrics, for analysis/rag_eval.py's decision_fn slot.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from decisions.client import DecisionClient
from decisions.types import ChoiceQ, NoulQ, ScoreQ

_MAX_CHUNKS_DEFAULT = 20

_RELEVANCE_LEVELS = [
    ("low", "the answer barely or does not address the question"),
    ("medium", "the answer partially addresses the question"),
    ("high", "the answer directly and completely addresses the question"),
]

_FAULT_CRITERIA = {
    "retriever": "Relevant information is missing from CONTEXT.",
    "generator": "CONTEXT is sufficient but ANSWER misuses or ignores it.",
    "both": "Both retrieval and generation are faulty.",
    "none": "No fault; ANSWER is correct and grounded in CONTEXT.",
}


def make_rag_decision_fn(
    client: DecisionClient,
    max_chunks: int = _MAX_CHUNKS_DEFAULT,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a decision_fn for evaluate_rag_case(decision_fn=...)/evaluate_rag_report.

    `case`: {"question": str, "contexts": list[str], "answer": str,
    "expected_answer": str (optional)}. One Jev call per case, regardless of
    chunk count (capped at max_chunks context-relevance questions).
    """

    def decide(case: Dict[str, Any]) -> Dict[str, Any]:
        question = case.get("question", "")
        contexts: List[str] = list(case.get("contexts") or [])[:max_chunks]
        answer = case.get("answer", "")
        expected_answer = case.get("expected_answer")

        state_parts = [f"QUESTION:\n{question}"]
        for i, chunk in enumerate(contexts):
            state_parts.append(f"CONTEXT [{i}]:\n{chunk}")
        state_parts.append(f"ANSWER:\n{answer}")
        if expected_answer:
            state_parts.append(f"EXPECTED ANSWER:\n{expected_answer}")
        state = "\n\n".join(state_parts)

        questions: Dict[str, Any] = {}
        for i in range(len(contexts)):
            questions[f"ctx_{i}"] = NoulQ(
                f"CONTEXT [{i}] contains information relevant to answering QUESTION."
            )
        questions["faithful"] = NoulQ(
            "Every factual claim in ANSWER is supported by the CONTEXT blocks."
        )
        questions["relevant"] = ScoreQ(
            "Rate how directly ANSWER addresses QUESTION.", _RELEVANCE_LEVELS
        )
        if expected_answer:
            questions["recall"] = NoulQ(
                "The CONTEXT blocks together contain the information needed to "
                "produce EXPECTED ANSWER."
            )
        questions["fault"] = ChoiceQ(
            "Where is the fault, if any, in this RAG case?", _FAULT_CRITERIA
        )

        result = client.decide(state, questions)
        answers = result.answers

        chunk_scores = [answers[f"ctx_{i}"].value for i in range(len(contexts))]
        relevant_count = sum(1 for s in chunk_scores if s >= 0.5)
        precision = relevant_count / len(contexts) if contexts else 0.0

        context_recall: Optional[Dict[str, Any]]
        if expected_answer and "recall" in answers:
            context_recall = {
                "recall": answers["recall"].value,
                "covered_tokens": None,
                "total_tokens": None,
            }
        else:
            context_recall = {"recall": None, "covered_tokens": None, "total_tokens": None}

        fault_answer = answers["fault"]

        return {
            "context_precision": {
                "precision": precision,
                "relevant_count": relevant_count,
                "total_chunks": len(contexts),
                "chunk_scores": chunk_scores,
            },
            "context_recall": context_recall,
            "faithfulness": {
                "faithfulness": answers["faithful"].value,
                "grounded_tokens": None,
                "total_answer_tokens": None,
            },
            "answer_relevance": {"answer_relevance": answers["relevant"].value},
            "decision_fault": {
                "fault": fault_answer.value,
                "confidence": fault_answer.confidence,
            },
        }

    return decide
