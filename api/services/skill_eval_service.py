"""Skill Quality Lab service — static lint + task-fit judge + report store."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from analysis.skill_lint import lint_skill
from evaluators.skill_fit_judge import SkillFitJudge
from utils.logger import get_logger

logger = get_logger(__name__)

REPORT_PREFIX = "skill_eval_"


def _default_adapter_factory(model_key: str, config_path: str) -> Any:
    """Build a UnifiedLLMAdapter for `model_key` with ${ENV_VAR} expansion."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    config = yaml.safe_load(config_str)
    if model_key not in config.get("models", {}):
        raise ValueError(f"Model '{model_key}' not found in config")
    from adapters.unified_adapter import UnifiedLLMAdapter  # heavy import kept lazy

    return UnifiedLLMAdapter(dict(config["models"][model_key]), model_key=model_key)


class SkillEvalService:
    def __init__(
        self,
        reports_dir: str = "reports",
        config_path: str = "config/models.yaml",
        adapter_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self.reports_dir = Path(reports_dir)
        self.config_path = config_path
        self.adapter_factory = adapter_factory or _default_adapter_factory

    # ── layers ────────────────────────────────────────────────────────────

    def lint(self, skill_text: str) -> Dict[str, Any]:
        return lint_skill(skill_text)

    def fit(self, skill_text: str, task_description: str, judge_model: str) -> Optional[Dict[str, Any]]:
        """Task-fit verdict, or None when the judge output is unusable.

        Raises ValueError for an unknown judge_model key.
        """
        adapter = self.adapter_factory(judge_model, self.config_path)
        return SkillFitJudge(adapter).evaluate(skill_text, task_description)

    def full(
        self,
        skill_text: str,
        task_description: str,
        judge_model: str,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Lint + fit with a combined 0-1 score; optionally persisted."""
        lint_report = self.lint(skill_text)
        fit_report = self.fit(skill_text, task_description, judge_model)

        lint_norm = round(lint_report["score"] / 100.0, 4)
        if fit_report is not None:
            combined = round(0.5 * lint_norm + 0.5 * fit_report["overall"], 4)
            basis = "lint+fit"
        else:
            combined = lint_norm
            basis = "lint_only"

        report = {
            "kind": "skill_eval",
            "timestamp": datetime.now().isoformat(),
            "judge_model": judge_model,
            "task_description": task_description,
            "combined_score": combined,
            "combined_basis": basis,
            "lint": lint_report,
            "fit": fit_report,
        }
        if save:
            report["report_path"] = self._save(report)
        return report

    # ── report store ──────────────────────────────────────────────────────

    def _save(self, report: Dict[str, Any]) -> str:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{REPORT_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[skill_eval] report saved: {path}")
        return str(path)

    def list_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.reports_dir.exists():
            return []
        paths = sorted(
            self.reports_dir.glob(f"{REPORT_PREFIX}*.json"),
            key=lambda p: p.name,
            reverse=True,
        )[:limit]
        summaries = []
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[skill_eval] unreadable report {path}: {e}")
                continue
            fit = data.get("fit") or {}
            summaries.append(
                {
                    "filename": path.name,
                    "timestamp": data.get("timestamp"),
                    "judge_model": data.get("judge_model"),
                    "combined_score": data.get("combined_score"),
                    "combined_basis": data.get("combined_basis"),
                    "lint_score": (data.get("lint") or {}).get("score"),
                    "fit_overall": fit.get("overall"),
                    "verdict": fit.get("verdict"),
                    "skill_name": ((data.get("lint") or {}).get("summary") or {}).get("name"),
                }
            )
        return summaries

    def get_report(self, filename: str) -> Optional[Dict[str, Any]]:
        # Basename-only to prevent path traversal out of reports_dir.
        path = self.reports_dir / Path(filename).name
        if not path.name.startswith(REPORT_PREFIX) or not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
