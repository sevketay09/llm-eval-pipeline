"""Classifier & Guardrail Bench service — runs decision/LLM classifiers (choice
mode) or a Jev-scored guardrail (guardrail mode) and persists JSON reports.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from api.schemas.classifier_bench import BenchDetail, BenchSummary, ClassifierResult, CreateBenchRequest
from decisions.bench import (
    make_decision_classifier,
    make_decision_guardrail_fn,
    make_llm_classifier,
    run_choice_bench,
    run_guardrail_bench,
)
from decisions.config import build_decision_client

REPORT_PREFIX = "classifier_bench_"
_CASE_WORKERS = 4


def _default_adapter_factory(model_key: str, config_path: str) -> Any:
    """Build a UnifiedLLMAdapter for `model_key` with ${ENV_VAR} expansion.

    Mirrors RedTeamService/SkillEvalService's factory.
    """
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


def _model_cost_per_mtok_input(model_key: str, config_path: str) -> Optional[float]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except OSError:
        return None
    return (config.get("models", {}).get(model_key) or {}).get("cost_per_mtok_input")


class ClassifierBenchService:
    def __init__(
        self,
        reports_dir: str = "reports",
        config_path: str = "config/models.yaml",
        adapter_factory: Optional[Callable[[str, str], Any]] = None,
        decision_client_factory: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.config_path = config_path
        self.adapter_factory = adapter_factory or _default_adapter_factory
        self.decision_client_factory = decision_client_factory or build_decision_client
        self._benches: Dict[str, Dict[str, Any]] = {}

    def create(self, request: CreateBenchRequest) -> Dict[str, Any]:
        bench_id = uuid.uuid4().hex[:12]
        bench: Dict[str, Any] = {
            "bench_id": bench_id,
            "name": request.name,
            "mode": request.mode,
            "status": "pending",
            "created_at": time.time(),
            "finished_at": None,
            "error": "",
            "request": request,
            "results": [],
        }
        self._benches[bench_id] = bench
        return bench

    def get(self, bench_id: str) -> Optional[Dict[str, Any]]:
        return self._benches.get(bench_id)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        return sorted(self._benches.values(), key=lambda b: b["created_at"], reverse=True)[:limit]

    def run(self, bench_id: str) -> Optional[Dict[str, Any]]:
        bench = self._benches.get(bench_id)
        if bench is None:
            return None
        if bench["status"] == "running":
            return bench

        bench["status"] = "running"
        request: CreateBenchRequest = bench["request"]
        try:
            bench["results"] = (
                self._run_choice_mode(request)
                if request.mode == "choice"
                else self._run_guardrail_mode(request)
            )
            bench["status"] = "done"
        except Exception as exc:  # noqa: BLE001 — surfaced as bench.error, not a crash
            bench["status"] = "error"
            bench["error"] = str(exc)
        finally:
            bench["finished_at"] = time.time()
            self._save(bench)
        return bench

    def _run_choice_mode(self, request: CreateBenchRequest) -> List[ClassifierResult]:
        cases = [{"id": c.id, "text": c.text, "label": c.label} for c in request.cases]
        results: List[ClassifierResult] = []

        try:
            client = self.decision_client_factory(request.decision_model, self.config_path)
            classify_fn = make_decision_classifier(client, request.criteria, request.instructions)
            metrics = run_choice_bench(
                request.decision_model, classify_fn, cases, request.repeats, max_workers=_CASE_WORKERS
            )
            results.append(
                ClassifierResult(
                    name=request.decision_model,
                    kind="decision",
                    model_key=request.decision_model,
                    metrics=metrics,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one classifier's failure must not sink the bench
            results.append(
                ClassifierResult(
                    name=request.decision_model, kind="decision", model_key=request.decision_model, error=str(exc)
                )
            )

        for model_key in request.llm_models:
            try:
                adapter = self.adapter_factory(model_key, self.config_path)
                cost_per_mtok = _model_cost_per_mtok_input(model_key, self.config_path)
                classify_fn = make_llm_classifier(
                    adapter.generate, request.criteria, request.instructions, cost_per_mtok_input=cost_per_mtok
                )
                metrics = run_choice_bench(
                    model_key, classify_fn, cases, request.repeats, max_workers=_CASE_WORKERS
                )
                results.append(ClassifierResult(name=model_key, kind="llm", model_key=model_key, metrics=metrics))
            except Exception as exc:  # noqa: BLE001
                results.append(ClassifierResult(name=model_key, kind="llm", model_key=model_key, error=str(exc)))

        return results

    def _run_guardrail_mode(self, request: CreateBenchRequest) -> List[ClassifierResult]:
        cases = [
            {"id": c.id, "text": c.text, "expected": c.expected_verdict, "labels": c.labels}
            for c in request.cases
        ]
        categories = {name: cfg.model_dump() for name, cfg in request.guardrail_categories.items()}
        thresholds = {
            name: {k: v for k, v in {"block": cfg.block, "review": cfg.review}.items() if v is not None}
            for name, cfg in request.guardrail_categories.items()
        }
        instructions = {name: cfg.instructions for name, cfg in request.guardrail_categories.items()}

        try:
            client = self.decision_client_factory(request.decision_model, self.config_path)
            decide_fn = make_decision_guardrail_fn(client, instructions)
            metrics = run_guardrail_bench(
                request.decision_model, decide_fn, cases, categories, thresholds, max_workers=_CASE_WORKERS
            )
            return [
                ClassifierResult(
                    name=request.decision_model, kind="decision", model_key=request.decision_model, metrics=metrics
                )
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                ClassifierResult(
                    name=request.decision_model, kind="decision", model_key=request.decision_model, error=str(exc)
                )
            ]

    def to_summary(self, bench: Dict[str, Any]) -> BenchSummary:
        request: CreateBenchRequest = bench["request"]
        return BenchSummary(
            bench_id=bench["bench_id"],
            name=bench["name"],
            mode=bench["mode"],
            status=bench["status"],
            case_count=len(request.cases),
            classifier_names=[r.name for r in bench["results"]],
            created_at=bench["created_at"],
            finished_at=bench["finished_at"],
            error=bench["error"],
        )

    def to_detail(self, bench: Dict[str, Any]) -> BenchDetail:
        request: CreateBenchRequest = bench["request"]
        summary = self.to_summary(bench)
        return BenchDetail(
            **summary.model_dump(),
            criteria=request.criteria,
            instructions=request.instructions,
            guardrail_categories=request.guardrail_categories,
            decision_model=request.decision_model,
            llm_models=request.llm_models,
            repeats=request.repeats,
            results=bench["results"],
        )

    def _save(self, bench: Dict[str, Any]) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{REPORT_PREFIX}{bench['bench_id']}.json"
        payload = {
            **{k: v for k, v in bench.items() if k not in ("request", "results")},
            "request": bench["request"].model_dump(),
            "results": [r.model_dump() for r in bench["results"]],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
