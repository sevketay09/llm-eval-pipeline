"""Jev-backed skill routing: single-skill trigger probability, and
multi-skill selection among many candidate SKILL.md files.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from decisions.client import DecisionClient
from decisions.types import ChoiceQ, NoulQ

NONE_LABEL = "none"
_MAX_CHOICE_OPTIONS = 200  # stay under the SDK's 255 cap with headroom for "none"


def trigger_probability(client: DecisionClient, meta: Dict[str, str], text: str) -> float:
    """P(this skill should fire for `text`), per a single-skill Noul call."""
    state = (
        f"SKILL NAME: {meta.get('name') or '(unnamed)'}\n"
        f"SKILL DESCRIPTION: {meta.get('description') or '(no description)'}\n\n"
        f"USER REQUEST:\n{text}"
    )
    question = NoulQ(
        "An agent that has only this skill should invoke it to handle USER "
        "REQUEST. Unrelated requests must not trigger it."
    )
    result = client.decide(state, {"trigger": question})
    return float(result.answers["trigger"].value)


def route(
    client: DecisionClient,
    skills: Dict[str, str],
    text: str,
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Pick the best-fitting skill (or "none") for `text` among `skills`
    (name -> description). Two-stage selection above `_MAX_CHOICE_OPTIONS`
    skills: chunk, take each chunk's winner, then choose among winners.
    """
    names = list(skills.keys())

    if len(names) <= _MAX_CHOICE_OPTIONS:
        return _choose(client, skills, text, top_k=top_k)

    finalists: Dict[str, str] = {}
    for i in range(0, len(names), _MAX_CHOICE_OPTIONS):
        chunk_names = names[i : i + _MAX_CHOICE_OPTIONS]
        chunk = {n: skills[n] for n in chunk_names}
        chunk_result = _choose(client, chunk, text, top_k=1)
        winner = chunk_result["choice"]
        if winner != NONE_LABEL:
            finalists[winner] = skills[winner]

    if not finalists:
        return {"choice": NONE_LABEL, "confidence": 1.0, "ranked": [(NONE_LABEL, 1.0)]}
    if len(finalists) == 1:
        (only_name,) = finalists.keys()
        return {"choice": only_name, "confidence": 1.0, "ranked": [(only_name, 1.0)]}
    return _choose(client, finalists, text, top_k=top_k)


def _choose(client: DecisionClient, skills: Dict[str, str], text: str, *, top_k: int) -> Dict[str, Any]:
    criteria = dict(skills)
    criteria[NONE_LABEL] = "No listed skill fits the request"
    question = ChoiceQ(
        "Which skill (if any) should handle USER REQUEST, based on the "
        "listed skill descriptions?",
        criteria,
    )
    result = client.decide(f"USER REQUEST:\n{text}", {"skill": question})
    answer = result.answers["skill"]
    ranked: List[Tuple[str, float]] = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
    return {"choice": answer.value, "confidence": answer.confidence, "ranked": ranked[:top_k]}


def routing_choice_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy, per-skill precision/recall, none-rate and confusion over a
    list of {"expected": name|"none", "predicted": name|"none"} results.
    """
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["expected"])
    none_predicted = sum(1 for r in results if r["predicted"] == NONE_LABEL)
    confusion: Dict[str, Counter] = {}
    for r in results:
        confusion.setdefault(r["expected"], Counter())[r["predicted"]] += 1

    labels = sorted({r["expected"] for r in results} | {r["predicted"] for r in results})
    per_skill: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        tp = sum(1 for r in results if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in results if r["expected"] != label and r["predicted"] == label)
        fn = sum(1 for r in results if r["expected"] == label and r["predicted"] != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        per_skill[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
        }

    return {
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "none_rate": round(none_predicted / total, 4) if total else None,
        "per_skill": per_skill,
        "confusion": {expected: dict(counts) for expected, counts in confusion.items()},
    }
