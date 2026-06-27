"""Service for loading/saving models.yaml configuration."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from api.config import get_settings


class ConfigService:
    """Thread-safe config file management with backup."""

    def __init__(self, config_path: str | None = None):
        settings = get_settings()
        self._path = Path(config_path or settings.models_config_path)
        self._tests_path = Path(settings.tests_config_path)
        self._task_registry_path = Path(settings.task_registry_path)

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_full_config(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_models(self) -> dict[str, Any]:
        return self.load_full_config().get("models", {})

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return self.get_models().get(model_id)

    def get_embedding_models(self) -> dict[str, Any]:
        return self.load_full_config().get("embedding_models", {})

    def get_test_suites(self) -> dict[str, Any]:
        if not self._tests_path.exists():
            return {}
        with open(self._tests_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("test_suites", {})

    def get_task_registry(self) -> dict[str, Any]:
        if not self._task_registry_path.exists():
            return {}
        with open(self._task_registry_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("tasks", {})

    # ── Write ────────────────────────────────────────────────────────────────

    def save_models(self, models: dict[str, Any]) -> bool:
        full = self.load_full_config()
        full["models"] = models
        return self._save_full(full)

    def upsert_model(self, model_id: str, config: dict[str, Any]) -> bool:
        models = self.get_models()
        models[model_id] = config
        return self.save_models(models)

    def delete_model(self, model_id: str) -> bool:
        models = self.get_models()
        if model_id not in models:
            return False
        del models[model_id]
        return self.save_models(models)

    def import_models(self, new_models: dict[str, Any], overwrite: bool = False) -> tuple[int, int]:
        """Import models. Returns (added, skipped)."""
        models = self.get_models()
        added, skipped = 0, 0
        for mid, mcfg in new_models.items():
            if not isinstance(mcfg, dict):
                continue
            if mid in models and not overwrite:
                skipped += 1
                continue
            models[mid] = mcfg
            added += 1
        self.save_models(models)
        return added, skipped

    # ── Internal ─────────────────────────────────────────────────────────────

    def _save_full(self, full_config: dict[str, Any]) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            backup = self._path.with_suffix(".yaml.backup")
            shutil.copy2(self._path, backup)

        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
