"""
experiments/runner.py — Lightweight experiment eval loop.
Injectable model_fn and score_fn — no real LLM needed in tests.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple

from experiments.store import Experiment, VariantResult


# ── Default scoring ───────────────────────────────────────────────────────────

def _fuzzy_score(output: str, expected: str) -> float:
    if not expected:
        return 1.0
    o = output.lower().strip()
    e = expected.lower().strip()
    if o == e:
        return 1.0
    if e in o:
        return 0.9
    return round(SequenceMatcher(None, o, e).ratio(), 4)


# ── Runner ────────────────────────────────────────────────────────────────────

ModelFn = Callable[[str, str], Tuple[str, float]]
"""(system_prompt, user_input) -> (output_text, latency_ms)"""

ScoreFn = Callable[[str, str], float]
"""(output, expected) -> score 0-1"""


class ExperimentRunner:
    def __init__(
        self,
        model_fn: ModelFn,
        score_fn: Optional[ScoreFn] = None,
    ):
        self._model_fn = model_fn
        self._score_fn = score_fn or _fuzzy_score

    def run(self, experiment: Experiment) -> List[VariantResult]:
        results: List[VariantResult] = []
        for variant in experiment.variants:
            for case in experiment.dataset:
                try:
                    output, latency_ms = self._model_fn(variant.system_prompt, case.input)
                    score = self._score_fn(output, case.expected)
                    results.append(
                        VariantResult(
                            variant_label=variant.label,
                            case_id=case.case_id,
                            output=output,
                            score=round(score, 4),
                            latency_ms=round(latency_ms, 2),
                        )
                    )
                except Exception as exc:
                    results.append(
                        VariantResult(
                            variant_label=variant.label,
                            case_id=case.case_id,
                            output="",
                            score=0.0,
                            latency_ms=0.0,
                            error=str(exc),
                        )
                    )
        return results
