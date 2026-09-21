"""Model config loading and decision-client factory.

`load_models_config` mirrors the `_default_adapter_factory` pattern used by
api/services/{redteam,rag_eval,custom_metric,skill_eval}_service.py: dump the
loaded YAML back to a string, substitute `${ENV_VAR}` placeholders, and
reload. ConfigService's own loader does not expand these placeholders, so
reading straight from it would hand a decision client a literal
"${TYPESAFE_API_KEY}" string.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

DECISION_PROVIDERS = {"typesafe", "typesafe-mock"}


def is_decision_provider(provider: Optional[str]) -> bool:
    return provider in DECISION_PROVIDERS


def load_models_config(config_path: str = "config/models.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    return yaml.safe_load(config_str) or {}


def get_model_config(model_key: str, config_path: str = "config/models.yaml") -> Dict[str, Any]:
    config = load_models_config(config_path)
    models = config.get("models", {})
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' not found in config")
    return dict(models[model_key])


def is_decision_model(model_key: str, config_path: str = "config/models.yaml") -> bool:
    try:
        model_cfg = get_model_config(model_key, config_path)
    except ValueError:
        return False
    return is_decision_provider(model_cfg.get("provider"))


def build_decision_client(model_key: str, config_path: str = "config/models.yaml") -> Any:
    model_cfg = get_model_config(model_key, config_path)
    provider = model_cfg.get("provider")

    if provider == "typesafe":
        from decisions.jev_client import JevDecisionClient

        return JevDecisionClient(model_cfg, model_key)
    if provider == "typesafe-mock":
        from decisions.mock_client import MockDecisionClient

        return MockDecisionClient(model_key, model_cfg.get("model_name") or "jev-mock")

    raise ValueError(f"Model '{model_key}' is not a decision model (provider={provider!r})")
