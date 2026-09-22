"""Jev-backed Dataset Studio quality control — one call per case to flag
ambiguous, nondeterministic, unsupported or easy/hard cases before a
generated or imported dataset ships.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from decisions.client import DecisionClient
from decisions.types import DecisionError, NoulQ, ScoreQ

_MAX_SOURCE_CHARS = 6000
_MIN_CASES_AFTER_DROP = 3

_DIFFICULTY_LEVELS = [
    ("easy", "a strong assistant answers this correctly with little effort"),
    ("medium", "a strong assistant needs real reasoning or domain knowledge to answer this"),
    ("hard", "even a strong assistant is likely to struggle with this"),
]


@dataclass(frozen=True)
class QCThresholds:
    ambiguous: float = 0.7
    nondeterministic: float = 0.7
    unsupported: float = 0.3


def _difficulty_label(value: float) -> str:
    n = len(_DIFFICULTY_LEVELS)
    idx = min(n - 1, max(0, round(value * (n - 1))))
    return _DIFFICULTY_LEVELS[idx][0]


def qc_cases(
    client: DecisionClient,
    cases: List[dict],
    *,
    source_material: Optional[str] = None,
    focus_areas: Optional[str] = None,
    thresholds: QCThresholds = QCThresholds(),
) -> Tuple[List[dict], dict]:
    """Annotate each case with `case["qc"]` and drop the ones that fail QC.

    Safety rule: if dropping would leave fewer than 3 cases, nothing is
    dropped — cases stay annotated only, and `qc_skipped_reason` is set.
    """
    source = (source_material or "").strip()[:_MAX_SOURCE_CHARS] or None

    qc_errors = 0
    difficulty_distribution: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    drop_reasons: Dict[int, str] = {}

    for index, case in enumerate(cases):
        question = str(case.get("question") or "")
        expected_answer = str(case.get("expected_answer") or "")

        state_parts = [f"QUESTION:\n{question}", f"EXPECTED ANSWER:\n{expected_answer}"]
        if source:
            state_parts.append(f"SOURCE:\n{source}")
        if focus_areas:
            state_parts.append(f"FOCUS AREAS:\n{focus_areas}")
        state = "\n\n".join(state_parts)

        questions: Dict[str, Any] = {
            "ambiguous": NoulQ("QUESTION admits more than one reasonable correct answer, or is unclear."),
            "nondeterministic": NoulQ(
                "EXPECTED ANSWER depends on time, opinion or unstated context, so it cannot be judged automatically."
            ),
            "difficulty": ScoreQ(
                "Rate how difficult QUESTION is for a strong assistant.", _DIFFICULTY_LEVELS
            ),
        }
        if source:
            questions["supported"] = NoulQ("EXPECTED ANSWER is consistent with and supported by SOURCE.")

        try:
            result = client.decide(state, questions)
        except DecisionError:
            qc_errors += 1
            case["qc"] = {"error": True}
            continue

        answers = result.answers
        ambiguous_p = answers["ambiguous"].value
        nondeterministic_p = answers["nondeterministic"].value
        supported_p = answers["supported"].value if "supported" in answers else None
        difficulty_score = answers["difficulty"].value
        difficulty_label = _difficulty_label(difficulty_score)

        case["qc"] = {
            "ambiguous": ambiguous_p,
            "nondeterministic": nondeterministic_p,
            "supported": supported_p,
            "difficulty": difficulty_label,
            "difficulty_score": difficulty_score,
        }
        difficulty_distribution[difficulty_label] += 1

        if ambiguous_p >= thresholds.ambiguous:
            drop_reasons[index] = "ambiguous"
        elif nondeterministic_p >= thresholds.nondeterministic:
            drop_reasons[index] = "nondeterministic"
        elif supported_p is not None and supported_p <= thresholds.unsupported:
            drop_reasons[index] = "unsupported"

    summary: Dict[str, Any] = {
        "qc_model": getattr(client, "model_key", None),
        "qc_ambiguous_removed": 0,
        "qc_nondeterministic_removed": 0,
        "qc_unsupported_removed": 0,
        "difficulty_distribution": difficulty_distribution,
        "qc_errors": qc_errors,
    }

    if drop_reasons and len(cases) - len(drop_reasons) < _MIN_CASES_AFTER_DROP:
        summary["qc_skipped_reason"] = "would_leave_fewer_than_3"
        return cases, summary

    kept_cases = []
    for index, case in enumerate(cases):
        reason = drop_reasons.get(index)
        if reason is None:
            kept_cases.append(case)
        else:
            summary[f"qc_{reason}_removed"] += 1

    return kept_cases, summary
