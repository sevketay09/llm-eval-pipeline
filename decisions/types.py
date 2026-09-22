"""Decision question/answer types — standalone, no imports from api/, utils/, adapters/, evaluators/."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

MAX_CHOICE_OPTIONS = 255


@dataclass(frozen=True)
class NoulQ:
    """A yes/no question. The answer is P(true) in [0, 1]."""

    instructions: str


@dataclass(frozen=True)
class ChoiceQ:
    """A question answered by selecting one of `criteria`'s labels."""

    instructions: str
    criteria: Dict[str, str]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("ChoiceQ requires at least one option")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"Choice supports at most {MAX_CHOICE_OPTIONS} options; use two-stage selection"
            )


@dataclass(frozen=True)
class ScoreQ:
    """A question scored against an ordered rubric, low to high.

    `levels` is a nonempty, ordered list of (name, description) pairs, e.g.
    [("low", "..."), ("medium", "..."), ("high", "...")]. The name is used
    only on this side (for readable probability keys); the wire format sends
    just the ordered descriptions.
    """

    instructions: str
    levels: List[Tuple[str, str]]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("ScoreQ requires at least one level")


Question = Union[NoulQ, ChoiceQ, ScoreQ]


@dataclass
class Decision:
    """One answered question."""

    kind: str  # "noul" | "choice" | "score"
    value: Union[float, str]  # noul: P(true) 0..1 | choice: label | score: 0..1 (normalized)
    confidence: float | None
    probabilities: Dict[str, float] = field(default_factory=dict)

    def as_float(self) -> float | None:
        """Numeric reading of this decision, regardless of kind."""
        if self.kind in ("noul", "score"):
            return float(self.value)  # type: ignore[arg-type]
        if self.kind == "choice":
            return self.probabilities.get(str(self.value))
        return None


@dataclass
class DecisionResponse:
    """Result of one `DecisionClient.decide()` call, covering every question asked."""

    answers: Dict[str, Decision]
    latency_ms: float
    est_input_tokens: int
    model: str


class DecisionError(RuntimeError):
    """Raised when a decision model call fails after any retries.

    Callers (evaluators, scorers) must catch this and fall back to `None`/a
    cascade fallback — never fabricate a decision on failure.
    """
