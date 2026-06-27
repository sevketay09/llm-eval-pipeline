"""Contract tests for experiments/runner.py — fake model, no LLM."""
from __future__ import annotations
import pytest
from experiments.store import ExperimentCase, Experiment, PromptVariant, make_experiment
from experiments.runner import ExperimentRunner, _fuzzy_score


def fake_model(system_prompt: str, user_input: str):
    return f"response to {user_input}", 42.0


def failing_model(system_prompt: str, user_input: str):
    raise RuntimeError("LLM unavailable")


def _exp(n_variants=2, n_cases=3) -> Experiment:
    return make_experiment(
        name="test",
        variants=[PromptVariant(label=f"v{i}", system_prompt=f"prompt {i}") for i in range(n_variants)],
        dataset=[ExperimentCase(case_id=f"c{i}", input=f"q{i}", expected=f"a{i}") for i in range(n_cases)],
    )


class TestFuzzyScore:
    def test_exact_match(self):
        assert _fuzzy_score("hello", "hello") == 1.0

    def test_contains(self):
        score = _fuzzy_score("the answer is 42", "42")
        assert score >= 0.9

    def test_empty_expected(self):
        assert _fuzzy_score("anything", "") == 1.0

    def test_mismatch_below_one(self):
        score = _fuzzy_score("completely different", "hello world")
        assert score < 1.0


class TestExperimentRunner:
    def test_run_produces_correct_count(self):
        exp = _exp(n_variants=2, n_cases=3)
        runner = ExperimentRunner(model_fn=fake_model)
        results = runner.run(exp)
        assert len(results) == 6  # 2 variants × 3 cases

    def test_result_has_variant_label(self):
        exp = _exp(n_variants=1, n_cases=1)
        runner = ExperimentRunner(model_fn=fake_model)
        results = runner.run(exp)
        assert results[0].variant_label == "v0"

    def test_result_has_case_id(self):
        exp = _exp(n_variants=1, n_cases=2)
        runner = ExperimentRunner(model_fn=fake_model)
        results = runner.run(exp)
        case_ids = {r.case_id for r in results}
        assert case_ids == {"c0", "c1"}

    def test_latency_from_model(self):
        exp = _exp(n_variants=1, n_cases=1)
        runner = ExperimentRunner(model_fn=fake_model)
        results = runner.run(exp)
        assert results[0].latency_ms == 42.0

    def test_failing_model_records_error(self):
        exp = _exp(n_variants=1, n_cases=1)
        runner = ExperimentRunner(model_fn=failing_model)
        results = runner.run(exp)
        assert results[0].error != ""
        assert results[0].score == 0.0

    def test_custom_score_fn(self):
        exp = _exp(n_variants=1, n_cases=1)
        runner = ExperimentRunner(model_fn=fake_model, score_fn=lambda o, e: 0.42)
        results = runner.run(exp)
        assert results[0].score == 0.42

    def test_empty_dataset_no_results(self):
        exp = make_experiment(
            name="empty",
            variants=[PromptVariant(label="v1", system_prompt="p")],
            dataset=[],
        )
        runner = ExperimentRunner(model_fn=fake_model)
        assert runner.run(exp) == []
