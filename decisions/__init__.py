"""decisions/ — standalone package for Jev (TypeSafe AI) decision-model
integration. No imports from api/, utils/, adapters/, or evaluators/;
tracing/ and redteam/ follow the same standalone pattern.
"""
from __future__ import annotations

from decisions.client import DecisionClient, DecisionStats
from decisions.config import build_decision_client, is_decision_model, is_decision_provider
from decisions.mock_client import MockDecisionClient
from decisions.types import (
    ChoiceQ,
    Decision,
    DecisionError,
    DecisionResponse,
    NoulQ,
    Question,
    ScoreQ,
)

__all__ = [
    "ChoiceQ",
    "Decision",
    "DecisionClient",
    "DecisionError",
    "DecisionResponse",
    "DecisionStats",
    "MockDecisionClient",
    "NoulQ",
    "Question",
    "ScoreQ",
    "build_decision_client",
    "is_decision_model",
    "is_decision_provider",
]
