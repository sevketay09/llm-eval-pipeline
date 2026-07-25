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

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptVariant":
        return cls(label=data["label"], system_prompt=data["system_prompt"], metadata=data.get("metadata", {}))


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

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentCase":
        return cls(
            case_id=data["case_id"],
            input=data["input"],
            expected=data.get("expected", ""),
            metadata=data.get("metadata", {}),
        )


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

    @classmethod
    def from_dict(cls, data: Dict) -> "VariantResult":
        return cls(
            variant_label=data["variant_label"],
            case_id=data["case_id"],
            output=data["output"],
            score=data["score"],
            latency_ms=data["latency_ms"],
            error=data.get("error", ""),
        )


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

    @classmethod
    def from_dict(cls, data: Dict) -> "Experiment":
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            variants=[PromptVariant.from_dict(v) for v in data.get("variants", [])],
            dataset=[ExperimentCase.from_dict(c) for c in data.get("dataset", [])],
            model_key=data.get("model_key", ""),
            results=[VariantResult.from_dict(r) for r in data.get("results", [])],
            status=data.get("status", "pending"),
            error=data.get("error", ""),
            created_at=data.get("created_at", time.time()),
            finished_at=data.get("finished_at"),
        )


_MAX_EXPERIMENTS = 500


class ExperimentStore:
    def __init__(self):
        self._store: Dict[str, Experiment] = {}
        self._order: List[str] = []

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

    def save(self, path: Path) -> None:
        """Snapshot current state to disk (atomic write) so a process restart doesn't lose it."""
        data = [self._store[eid].to_dict() for eid in self._order if eid in self._store]
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)

    def load_from(self, path: Path) -> None:
        """Replace current state with a previously saved snapshot, if one exists."""
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        self._store = {}
        self._order = []
        for item in raw:
            exp = Experiment.from_dict(item)
            self._order.append(exp.experiment_id)
            self._store[exp.experiment_id] = exp


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
