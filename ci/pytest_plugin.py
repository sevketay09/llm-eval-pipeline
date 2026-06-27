"""pytest helpers for gating LLM evaluations inside a normal test suite.

Lets developers assert evaluation quality the same way they assert anything
else, so eval lives in the dev/CI loop. Import the helpers directly:

    from ci.pytest_plugin import (
        load_report, assert_weighted_score, assert_test_score,
        assert_no_regression, assert_gate,
    )

    def test_quality():
        report = load_report("reports/latest.json")
        assert_weighted_score(report, "gpt-4o", min_score=0.80)
        assert_test_score(report, "gpt-4o", "function_calling", min_score=0.80)

    def test_no_regression(baseline_report):
        report = load_report("reports/latest.json")
        assert_no_regression(report, baseline_report, "gpt-4o", max_drop=0.03)

These are plain assertion helpers (no plugin registration needed). They raise
``AssertionError`` with a readable message so pytest reports them naturally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ci.gate import (
    evaluate_gate,
    load_gate_config,
    load_report,
    _weighted_score,
    _test_overall_score,
)

__all__ = [
    "load_report",
    "assert_weighted_score",
    "assert_test_score",
    "assert_no_regression",
    "assert_gate",
]


def _model(report: dict[str, Any], model_key: str) -> dict[str, Any]:
    model = (report.get("models", {}) or {}).get(model_key)
    assert model is not None, f"model '{model_key}' not found in report"
    return model


def assert_weighted_score(report: dict[str, Any], model_key: str, min_score: float) -> None:
    """Assert a model's weighted_score meets a minimum."""
    val = _weighted_score(_model(report, model_key))
    assert val is not None, f"{model_key}: weighted_score missing"
    assert val >= min_score, (
        f"{model_key}: weighted_score {val:.4f} < required {min_score}"
    )


def assert_test_score(
    report: dict[str, Any], model_key: str, test_name: str, min_score: float
) -> None:
    """Assert a specific test's overall_score meets a minimum."""
    test_data = (_model(report, model_key).get("tests", {}) or {}).get(test_name)
    assert isinstance(test_data, dict), f"{model_key}: test '{test_name}' not found"
    assert not test_data.get("error"), (
        f"{model_key}.{test_name}: errored — {test_data.get('error')}"
    )
    val = _test_overall_score(test_data)
    assert val is not None, f"{model_key}.{test_name}: overall_score missing"
    assert val >= min_score, (
        f"{model_key}.{test_name}: overall_score {val:.4f} < required {min_score}"
    )


def assert_no_regression(
    report: dict[str, Any],
    baseline: dict[str, Any],
    model_key: str,
    max_drop: float = 0.03,
    test_name: Optional[str] = None,
) -> None:
    """Assert a model has not regressed beyond ``max_drop`` vs baseline.

    When ``test_name`` is given, compares that test's overall_score; otherwise
    compares the model's weighted_score.
    """
    model = _model(report, model_key)
    base_model = _model(baseline, model_key)
    if test_name:
        cur = _test_overall_score((model.get("tests", {}) or {}).get(test_name, {}))
        base = _test_overall_score((base_model.get("tests", {}) or {}).get(test_name, {}))
        label = f"{model_key}.{test_name} overall_score"
    else:
        cur, base = _weighted_score(model), _weighted_score(base_model)
        label = f"{model_key} weighted_score"
    assert cur is not None and base is not None, f"{label}: missing current/baseline value"
    delta = cur - base
    assert delta >= -max_drop, (
        f"{label} regressed by {abs(delta):.4f} "
        f"({base:.4f} -> {cur:.4f}); allowed drop {max_drop}"
    )


def assert_gate(
    report: dict[str, Any],
    config_path: str | Path = "config/ci_gate.yaml",
    baseline: Optional[dict[str, Any]] = None,
) -> None:
    """Assert the full configured gate passes (mirrors ``python -m ci.gate``)."""
    gate = evaluate_gate(report, load_gate_config(config_path), baseline=baseline)
    if not gate.passed:
        failures = []
        for m in gate.models:
            for c in m.failed_checks:
                failures.append(f"  {m.model_key} / {c.name}: {c.detail or 'failed'}")
        raise AssertionError("CI gate failed:\n" + "\n".join(failures))
