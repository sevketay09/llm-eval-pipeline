"""Experiment service — orchestrates prompt variant runs and diffs."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple

from api.schemas.experiments import (
    CompareResponse,
    ExperimentDetail,
    ExperimentSummary,
)
from experiments.differ import compute_diff
from experiments.runner import ExperimentRunner
from experiments.store import (
    Experiment,
    ExperimentCase,
    ExperimentStore,
    PromptVariant,
    VariantResult,
    make_experiment,
)


def _noop_model_fn(system_prompt: str, user_input: str) -> Tuple[str, float]:
    """Placeholder — real calls go through UnifiedLLMAdapter injection."""
    return f"[no model configured] prompt={system_prompt[:30]!r} input={user_input[:30]!r}", 0.0


class ExperimentService:
    def __init__(self, store: Optional[ExperimentStore] = None):
        self._store = store or ExperimentStore()
        self._lock = asyncio.Lock()

    def create(
        self,
        name: str,
        variants: List[PromptVariant],
        dataset: List[ExperimentCase],
        model_key: str = "",
    ) -> Experiment:
        exp = make_experiment(name=name, variants=variants, dataset=dataset, model_key=model_key)
        self._store.create(exp)
        return exp

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._store.get(experiment_id)

    def list(self, limit: int = 50) -> List[Experiment]:
        return self._store.list(limit=limit)

    def save_state(self, path: Path) -> None:
        self._store.save(path)

    def load_state(self, path: Path) -> None:
        self._store.load_from(path)

    async def run(
        self,
        experiment_id: str,
        model_fn=None,
    ) -> Optional[Experiment]:
        async with self._lock:
            exp = self._store.get(experiment_id)
            if exp is None:
                return None
            if exp.status == "running":
                return exp
            exp.status = "running"
            self._store.update(exp)

        fn = model_fn or _noop_model_fn
        runner = ExperimentRunner(model_fn=fn)
        try:
            results: List[VariantResult] = await asyncio.get_event_loop().run_in_executor(
                None, runner.run, exp
            )
            exp.results = results
            exp.status = "done"
        except Exception as exc:
            exp.status = "error"
            exp.error = str(exc)
        finally:
            exp.finished_at = time.time()
            self._store.update(exp)

        return exp

    def compare(
        self,
        experiment_id: str,
        base_variant: Optional[str] = None,
        compare_variant: Optional[str] = None,
    ) -> Optional[CompareResponse]:
        exp = self._store.get(experiment_id)
        if exp is None or not exp.results:
            return None

        labels = [v.label for v in exp.variants]
        base_label = base_variant or (labels[0] if labels else "")
        compare_label = compare_variant or (labels[1] if len(labels) > 1 else base_label)

        base_res = [r for r in exp.results if r.variant_label == base_label]
        compare_res = [r for r in exp.results if r.variant_label == compare_label]

        diffs = compute_diff(base_res, compare_res, base_label=base_label, compare_label=compare_label)

        counts = {"improved": 0, "regressed": 0, "stable": 0, "missing": 0}
        for d in diffs:
            counts[d.verdict] = counts.get(d.verdict, 0) + 1

        non_missing = [d for d in diffs if d.verdict != "missing"]
        avg_delta = round(sum(d.delta for d in non_missing) / len(non_missing), 4) if non_missing else 0.0

        from api.schemas.experiments import CaseDiffSchema
        return CompareResponse(
            experiment_id=experiment_id,
            base_label=base_label,
            compare_label=compare_label,
            diffs=[CaseDiffSchema(**d.to_dict()) for d in diffs],
            improved=counts["improved"],
            regressed=counts["regressed"],
            stable=counts["stable"],
            missing=counts["missing"],
            avg_delta=avg_delta,
        )

    def to_summary(self, exp: Experiment) -> ExperimentSummary:
        return ExperimentSummary(
            experiment_id=exp.experiment_id,
            name=exp.name,
            model_key=exp.model_key,
            status=exp.status,
            variant_count=len(exp.variants),
            case_count=len(exp.dataset),
            created_at=exp.created_at,
            finished_at=exp.finished_at,
        )

    def to_detail(self, exp: Experiment) -> ExperimentDetail:
        from api.schemas.experiments import (
            ExperimentCaseSchema,
            PromptVariantSchema,
            VariantResultSchema,
        )
        return ExperimentDetail(
            experiment_id=exp.experiment_id,
            name=exp.name,
            model_key=exp.model_key,
            variants=[PromptVariantSchema(**v.to_dict()) for v in exp.variants],
            dataset=[ExperimentCaseSchema(**c.to_dict()) for c in exp.dataset],
            results=[VariantResultSchema(**r.to_dict()) for r in exp.results],
            status=exp.status,
            error=exp.error,
            created_at=exp.created_at,
            finished_at=exp.finished_at,
        )
