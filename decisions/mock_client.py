"""Deterministic, offline decision client — used for tests, demos, and any
run without a Jev API key. Same hash-bucket approach as tracing/sampler.py.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

from decisions.client import BaseDecisionClient
from decisions.types import ChoiceQ, Decision, DecisionResponse, NoulQ, Question, ScoreQ


def _hash_unit(*parts: str) -> float:
    """Deterministic value in [0, 1) from the given parts."""
    key = "|".join(parts).encode()
    h = int(hashlib.md5(key).hexdigest(), 16)
    return (h % 10_000) / 10_000


class MockDecisionClient(BaseDecisionClient):
    """Offline stand-in for JevDecisionClient. Same (state, question name) always
    yields the same answer, so contract tests stay deterministic.
    """

    def __init__(
        self,
        model_key: str = "typesafe-mock",
        model_name: str = "jev-mock",
        *,
        overrides: Dict[str, Any] | None = None,
        fail_times: int = 0,
    ) -> None:
        super().__init__(model_key, model_name)
        self._overrides = overrides or {}
        self._fail_times = fail_times
        self._calls = 0

    def _decide_raw(self, state: str, questions: Dict[str, Question]) -> DecisionResponse:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError(f"MockDecisionClient: simulated failure ({self._calls}/{self._fail_times})")

        answers: Dict[str, Decision] = {}
        for name, q in questions.items():
            override = self._overrides.get(name)
            answers[name] = _mock_decision(state, name, q, override)

        return DecisionResponse(
            answers=answers,
            latency_ms=1.0,
            est_input_tokens=max(1, len(state) // 4),
            model=self.model_name,
        )


def _mock_decision(state: str, name: str, q: Question, override: Any) -> Decision:
    h = _hash_unit(state, name)

    if isinstance(q, NoulQ):
        p = override if override is not None else 0.05 + 0.9 * h
        p = float(p)
        return Decision(kind="noul", value=p, confidence=max(p, 1 - p), probabilities={"true": p, "false": 1 - p})

    if isinstance(q, ChoiceQ):
        labels = list(q.criteria.keys())
        chosen = override if override is not None else labels[int(h * len(labels)) % len(labels)]
        remaining = [l for l in labels if l != chosen]
        probabilities = {chosen: 0.8}
        if remaining:
            share = 0.2 / len(remaining)
            probabilities.update({l: share for l in remaining})
        return Decision(kind="choice", value=chosen, confidence=probabilities[chosen], probabilities=probabilities)

    if isinstance(q, ScoreQ):
        v = override if override is not None else h
        v = float(v)
        return Decision(kind="score", value=v, confidence=max(v, 1 - v), probabilities={})

    raise ValueError(f"Unsupported question type for '{name}': {type(q).__name__}")
