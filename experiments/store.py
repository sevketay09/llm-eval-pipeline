"""
experiments/store.py — Prompt experiment dataclasses + in-memory store.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PromptVariant:
    label: str
    system_prompt: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"label": self.label, "system_prompt": self.system_prompt, "metadata": self.metadata}


@dataclass
class ExperimentCase:
    case_id: str
    input: str
    expected: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }


@dataclass
class VariantResult:
    variant_label: str
    case_id: str
    output: str
    score: float
    latency_ms: float
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "variant_label": self.variant_label,
            "case_id": self.case_id,
            "output": self.output,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class Experiment:
    experiment_id: str
    name: str
    variants: List[PromptVariant]
    dataset: List[ExperimentCase]
    model_key: str = ""
    results: List[VariantResult] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | error
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_key": self.model_key,
            "variants": [v.to_dict() for v in self.variants],
            "dataset": [c.to_dict() for c in self.dataset],
            "results": [r.to_dict() for r in self.results],
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_MAX_EXPERIMENTS = 500


class ExperimentStore:
    def __init__(self, dump_path: Optional[Path] = None):
        self._store: Dict[str, Experiment] = {}
        self._order: List[str] = []
        self._dump_path = dump_path

    def create(self, experiment: Experiment) -> Experiment:
        if experiment.experiment_id not in self._store:
            self._order.append(experiment.experiment_id)
        self._store[experiment.experiment_id] = experiment
        while len(self._order) > _MAX_EXPERIMENTS:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        return experiment

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._store.get(experiment_id)

    def list(self, limit: int = 50) -> List[Experiment]:
        return [self._store[eid] for eid in self._order[-limit:] if eid in self._store]

    def update(self, experiment: Experiment) -> None:
        self._store[experiment.experiment_id] = experiment

    def delete(self, experiment_id: str) -> bool:
        if experiment_id not in self._store:
            return False
        del self._store[experiment_id]
        if experiment_id in self._order:
            self._order.remove(experiment_id)
        return True

    def count(self) -> int:
        return len(self._store)

    def dump(self) -> None:
        if not self._dump_path:
            return
        data = [exp.to_dict() for exp in self._store.values()]
        self._dump_path.write_text(json.dumps(data, indent=2))


def make_experiment(
    name: str,
    variants: List[PromptVariant],
    dataset: List[ExperimentCase],
    model_key: str = "",
    experiment_id: Optional[str] = None,
) -> Experiment:
    return Experiment(
        experiment_id=experiment_id or uuid.uuid4().hex,
        name=name,
        variants=variants,
        dataset=dataset,
        model_key=model_key,
    )
