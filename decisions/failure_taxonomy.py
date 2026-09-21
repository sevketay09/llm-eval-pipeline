"""Jev-backed failure taxonomy classification for analysis/failure_clustering.py.

Standalone: takes plain failure dicts (model, test, case_id, score, category,
text, error), as produced by analysis.failure_clustering.extract_failures.
Does not import analysis/ (import direction is analysis/ -> decisions/).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from decisions.client import DecisionClient
from decisions.types import ChoiceQ, DecisionError

DEFAULT_FAILURE_TAXONOMY: Dict[str, str] = {
    "hallucination": "The answer states fabricated or unsupported facts.",
    "wrong_answer": "The answer is factually or logically wrong but not fabricated.",
    "refusal_overcautious": "The model refused or deflected a legitimate request.",
    "unsafe_compliance": "The model complied with an unsafe or disallowed request.",
    "wrong_tool": "Wrong tool, wrong arguments, or missing tool call.",
    "format_violation": "Output violates the required format/schema/JSON.",
    "reasoning_error": "Multi-step reasoning or calculation went wrong.",
    "retrieval_miss": "The needed information was not in the retrieved context.",
    "language_error": "Wrong language, broken Turkish, or language mixing.",
    "incomplete_answer": "The answer is partial or stops early.",
    "infra_error": "Timeout, API error or empty response — not a model quality issue.",
    "other": "None of the above.",
}


def make_classify_fn(
    client: DecisionClient,
    taxonomy: Optional[Dict[str, str]] = None,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build a classify_fn for compute_failure_summary(classify_fn=...).

    One Jev call per failure (skipped, for free, when the failure already
    carries an infra error). Returns [{"label", "confidence"}, ...] in the
    same order as the input list.
    """
    labels = taxonomy or DEFAULT_FAILURE_TAXONOMY
    question = ChoiceQ("Which category best describes this evaluation failure?", labels)

    def classify(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for failure in failures:
            if failure.get("error"):
                results.append({"label": "infra_error", "confidence": 1.0})
                continue

            state = (
                f"TEST: {failure.get('test', 'unknown')}\n"
                f"CATEGORY: {failure.get('category', 'unknown')}\n"
                f"SCORE: {failure.get('score')}\n\n"
                f"{failure.get('text', '')}"
            )
            try:
                resp = client.decide(state, {"category": question})
                answer = resp.answers["category"]
                results.append({"label": answer.value, "confidence": answer.confidence})
            except DecisionError:
                results.append({"label": "unclassified", "confidence": None})
        return results

    return classify
