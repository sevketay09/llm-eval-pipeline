"""Skill Quality Lab service — static lint + task-fit judge + report store."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from analysis.skill_lint import lint_skill
from utils.logger import get_logger

logger = get_logger(__name__)

REPORT_PREFIX = "skill_eval_"


class UnsupportedJudgeError(ValueError):
    """A decision (Jev) model was given where an LLM judge is required."""


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
        decision_client_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self.reports_dir = Path(reports_dir)
        self.config_path = config_path
        self.adapter_factory = adapter_factory or _default_adapter_factory
        from decisions.config import build_decision_client

        self.decision_client_factory = decision_client_factory or build_decision_client

    def _is_decision_model(self, model_key: str) -> bool:
        from decisions.config import is_decision_model

        return is_decision_model(model_key, self.config_path)

    # ── layers ────────────────────────────────────────────────────────────

    def lint(self, skill_text: str) -> Dict[str, Any]:
        return lint_skill(skill_text)

    def fit(self, skill_text: str, task_description: str, judge_model: str) -> Optional[Dict[str, Any]]:
        """Task-fit verdict, or None when the judge output is unusable.

        Raises ValueError for an unknown judge_model key, or for a decision
        (Jev) model — task-fit needs a judge that can explain its verdict.
        """
        if self._is_decision_model(judge_model):
            raise UnsupportedJudgeError("Skill fit requires an LLM judge, not a decision model")

        # Lazy import: pulling evaluators/__init__ drags in scipy-dependent
        # modules, which must not be a requirement for the lint-only path.
        from evaluators.skill_fit_judge import SkillFitJudge

        adapter = self.adapter_factory(judge_model, self.config_path)
        return SkillFitJudge(adapter).evaluate(skill_text, task_description)

    def trigger(
        self,
        skill_text: str,
        prompts: List[Dict[str, Any]],
        judge_model: str,
        repeats: int = 1,
    ) -> Dict[str, Any]:
        """Routing precision/recall report for a labeled prompt set.

        Raises ValueError for an unknown judge_model key.
        """
        # Lazy import for the same reason as fit(): keep the lint-only
        # path free of the scipy-dependent evaluators/__init__ chain.
        from analysis.skill_trigger import SkillTriggerChecker

        if self._is_decision_model(judge_model):
            client = self.decision_client_factory(judge_model, self.config_path)
            return SkillTriggerChecker(decision_client=client).run(skill_text, prompts)

        adapter = self.adapter_factory(judge_model, self.config_path)
        return SkillTriggerChecker(adapter, repeats=repeats).run(skill_text, prompts)

    def route(
        self,
        skills: List[Dict[str, str]],
        prompts: List[Dict[str, Any]],
        decision_model: str,
    ) -> Dict[str, Any]:
        """Multi-skill routing: which SKILL.md (if any) should handle each
        prompt. `skills`: [{"name", "skill_text"}, ...]. `prompts`:
        [{"text", "expected": skill_name | "none"}, ...].
        """
        from analysis.skill_trigger import extract_skill_meta
        from decisions.skill_routing import NONE_LABEL, route as decision_route, routing_choice_metrics
        from decisions.types import DecisionError

        client = self.decision_client_factory(decision_model, self.config_path)

        catalog: Dict[str, str] = {}
        for skill in skills:
            meta = extract_skill_meta(skill.get("skill_text") or "")
            name = (skill.get("name") or meta["name"] or "").strip()
            if not name:
                continue
            catalog[name] = meta["description"] or name

        results = []
        for prompt in prompts:
            text = str(prompt.get("text") or "").strip()
            expected = prompt.get("expected")
            if not text or not expected:
                continue
            try:
                outcome = decision_route(client, catalog, text)
            except DecisionError as e:
                logger.warning(f"[skill_route] probe failed for {text[:60]!r}: {e}")
                continue
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": outcome["choice"],
                    "confidence": outcome["confidence"],
                    "ranked": outcome["ranked"],
                }
            )

        return {
            "skills": list(catalog.keys()),
            "metrics": routing_choice_metrics(results) if results else {"total": 0},
            "results": results,
            "none_label": NONE_LABEL,
        }

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
