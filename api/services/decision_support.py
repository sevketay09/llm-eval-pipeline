"""Shared helpers services use to work with decision (Jev) models."""
from __future__ import annotations

from typing import Any, Dict, List

from decisions.config import build_decision_client, is_decision_model


def resolve_decision_client(model_key: str, config_path: str, cache: Dict[str, Any]) -> Any:
    """Build (or reuse) a decision client for `model_key`, cached by key.

    Services are process-lifetime singletons, so callers keep `cache` as an
    instance attribute rather than rebuilding a client on every request.
    """
    if model_key not in cache:
        cache[model_key] = build_decision_client(model_key, config_path)
    return cache[model_key]


def require_decision_model(model_key: str, config_path: str) -> None:
    if not model_key or not is_decision_model(model_key, config_path):
        raise ValueError(f"'{model_key}' is not a decision (Jev) model")


def ensure_not_decision_model(model_keys: List[str], config_path: str) -> None:
    decision_keys = [k for k in model_keys if is_decision_model(k, config_path)]
    if decision_keys:
        raise ValueError(f"Decision models cannot be evaluated as a target model: {decision_keys}")
