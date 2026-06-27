"""
experiments/differ.py — Case-level diff between two prompt variants.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from experiments.store import VariantResult

_REGRESSION_THRESHOLD = 0.05   # delta < -0.05 → regressed
_IMPROVEMENT_THRESHOLD = 0.05  # delta > +0.05 → improved


@dataclass
class CaseDiff:
    case_id: str
    base_label: str
    compare_label: str
    base_score: float
    compare_score: float
    base_output: str
    compare_output: str
    delta: float          # compare - base
    verdict: str          # improved | regressed | stable | missing

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "base_label": self.base_label,
            "compare_label": self.compare_label,
            "base_score": self.base_score,
            "compare_score": self.compare_score,
            "base_output": self.base_output,
            "compare_output": self.compare_output,
            "delta": self.delta,
            "verdict": self.verdict,
        }


def compute_diff(
    base_results: List[VariantResult],
    compare_results: List[VariantResult],
    base_label: str = "",
    compare_label: str = "",
) -> List[CaseDiff]:
    """Diff two variant result lists by case_id."""
    base_map: Dict[str, VariantResult] = {r.case_id: r for r in base_results}
    compare_map: Dict[str, VariantResult] = {r.case_id: r for r in compare_results}

    all_case_ids = sorted(set(base_map) | set(compare_map))
    diffs: List[CaseDiff] = []

    for case_id in all_case_ids:
        b = base_map.get(case_id)
        c = compare_map.get(case_id)

        if b is None or c is None:
            diffs.append(
                CaseDiff(
                    case_id=case_id,
                    base_label=base_label or (b.variant_label if b else ""),
                    compare_label=compare_label or (c.variant_label if c else ""),
                    base_score=b.score if b else 0.0,
                    compare_score=c.score if c else 0.0,
                    base_output=b.output if b else "",
                    compare_output=c.output if c else "",
                    delta=0.0,
                    verdict="missing",
                )
            )
            continue

        delta = round(c.score - b.score, 4)
        if delta > _IMPROVEMENT_THRESHOLD:
            verdict = "improved"
        elif delta < -_REGRESSION_THRESHOLD:
            verdict = "regressed"
        else:
            verdict = "stable"

        diffs.append(
            CaseDiff(
                case_id=case_id,
                base_label=base_label or b.variant_label,
                compare_label=compare_label or c.variant_label,
                base_score=b.score,
                compare_score=c.score,
                base_output=b.output,
                compare_output=c.output,
                delta=delta,
                verdict=verdict,
            )
        )

    return sorted(diffs, key=lambda d: d.delta)
