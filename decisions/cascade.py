"""Cascade policy — when a decision-model answer is confident enough to accept
on its own, versus when it should escalate to an LLM fallback or, downstream,
to human review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class CascadePolicy:
    accept_confidence: float = 0.85
    hitl_below: float = 0.60

    @classmethod
    def from_config(cls, judge_cfg: Optional[Dict[str, Any]]) -> "CascadePolicy":
        cascade_cfg = (judge_cfg or {}).get("cascade") or {}
        return cls(
            accept_confidence=float(cascade_cfg.get("accept_confidence", cls.accept_confidence)),
            hitl_below=float(cascade_cfg.get("hitl_below", cls.hitl_below)),
        )


def should_escalate(policy: CascadePolicy, confidences: Iterable[Optional[float]]) -> bool:
    """True if any confidence is missing or below the accept threshold."""
    values = list(confidences)
    if not values:
        return True
    return any(c is None or c < policy.accept_confidence for c in values)
