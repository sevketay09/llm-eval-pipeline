"""Contract tests for experiments/store.py — no LLM, no network."""
from __future__ import annotations
import time
import pytest
from experiments.store import (
    Experiment, ExperimentCase, ExperimentStore, PromptVariant,
    VariantResult, make_experiment, _MAX_EXPERIMENTS,
)


def _variant(label="v1", prompt="You are helpful.") -> PromptVariant:
    return PromptVariant(label=label, system_prompt=prompt)


def _case(case_id="c1", inp="hello", expected="hi") -> ExperimentCase:
    return ExperimentCase(case_id=case_id, input=inp, expected=expected)


def _exp(name="test", variant_labels=("v1", "v2"), n_cases=2) -> Experiment:
    return make_experiment(
        name=name,
        variants=[_variant(label=l) for l in variant_labels],
        dataset=[_case(case_id=f"c{i}") for i in range(n_cases)],
    )


class TestPromptVariant:
    def test_to_dict(self):
        v = _variant()
        d = v.to_dict()
        assert d["label"] == "v1"
        assert "system_prompt" in d

    def test_metadata_default_empty(self):
        v = _variant()
        assert v.metadata == {}


class TestExperimentCase:
    def test_to_dict(self):
        c = _case()
        d = c.to_dict()
        assert d["case_id"] == "c1"
        assert d["input"] == "hello"

    def test_expected_default_empty(self):
        c = ExperimentCase(case_id="x", input="q")
        assert c.expected == ""


class TestVariantResult:
    def test_to_dict(self):
        r = VariantResult(variant_label="v1", case_id="c1", output="ok", score=0.9, latency_ms=50.0)
        d = r.to_dict()
        assert d["score"] == 0.9
        assert d["error"] == ""


class TestExperiment:
    def test_to_dict_structure(self):
        exp = _exp()
        d = exp.to_dict()
        assert "experiment_id" in d
        assert d["status"] == "pending"
        assert len(d["variants"]) == 2
        assert len(d["dataset"]) == 2
        assert d["results"] == []

    def test_created_at_set(self):
        before = time.time()
        exp = _exp()
        assert exp.created_at >= before


class TestMakeExperiment:
    def test_generates_id(self):
        exp = _exp()
        assert exp.experiment_id
        assert len(exp.experiment_id) > 8

    def test_custom_id(self):
        exp = make_experiment("x", [_variant()], [_case()], experiment_id="custom-123")
        assert exp.experiment_id == "custom-123"


class TestExperimentStore:
    def test_create_and_get(self):
        store = ExperimentStore()
        exp = _exp()
        store.create(exp)
        assert store.get(exp.experiment_id) is exp

    def test_get_missing_none(self):
        store = ExperimentStore()
        assert store.get("nonexistent") is None

    def test_list(self):
        store = ExperimentStore()
        for i in range(3):
            store.create(_exp(name=f"exp-{i}"))
        assert store.count() == 3
        assert len(store.list()) == 3

    def test_list_limit(self):
        store = ExperimentStore()
        for i in range(10):
            store.create(_exp(name=f"e{i}"))
        assert len(store.list(limit=3)) == 3

    def test_update(self):
        store = ExperimentStore()
        exp = _exp()
        store.create(exp)
        exp.status = "done"
        store.update(exp)
        assert store.get(exp.experiment_id).status == "done"

    def test_delete(self):
        store = ExperimentStore()
        exp = _exp()
        store.create(exp)
        assert store.delete(exp.experiment_id) is True
        assert store.count() == 0

    def test_delete_missing_false(self):
        store = ExperimentStore()
        assert store.delete("nope") is False

    def test_fifo_eviction(self):
        store = ExperimentStore()
        first_id = "first"
        store.create(make_experiment("first", [_variant()], [_case()], experiment_id=first_id))
        for i in range(_MAX_EXPERIMENTS + 5):
            store.create(_exp(name=f"e{i}"))
        assert store.count() <= _MAX_EXPERIMENTS
        assert store.get(first_id) is None


class TestExperimentStorePersistence:
    def test_save_and_load_from_round_trips(self, tmp_path):
        path = tmp_path / "experiments.json"
        store = ExperimentStore()
        exp = _exp(name="persisted")
        exp.results = [VariantResult(variant_label="v1", case_id="c0", output="out", score=0.8, latency_ms=12.3)]
        store.create(exp)
        store.save(path)

        reloaded = ExperimentStore()
        reloaded.load_from(path)

        restored = reloaded.get(exp.experiment_id)
        assert restored is not None
        assert restored.name == "persisted"
        assert restored.results[0].output == "out"
        assert reloaded.list()[0].experiment_id == exp.experiment_id

    def test_load_from_missing_file_is_noop(self, tmp_path):
        store = ExperimentStore()
        store.load_from(tmp_path / "does-not-exist.json")
        assert store.count() == 0

    def test_save_preserves_insertion_order(self, tmp_path):
        path = tmp_path / "experiments.json"
        store = ExperimentStore()
        ids = []
        for i in range(3):
            exp = _exp(name=f"e{i}")
            ids.append(exp.experiment_id)
            store.create(exp)
        store.save(path)

        reloaded = ExperimentStore()
        reloaded.load_from(path)
        assert [e.experiment_id for e in reloaded.list()] == ids
