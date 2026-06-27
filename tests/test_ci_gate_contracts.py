"""Contract tests for the CI gate (ci/gate.py + ci/pytest_plugin.py).

Uses synthetic report payloads so the gate logic is verified without running
a real evaluation. Covers threshold checks, test-error gating, regression vs
baseline, renderers, and the pytest assertion helpers.
"""
import json

import pytest

from ci.gate import (
    evaluate_gate,
    render_markdown,
    render_badge,
    render_json,
    _normalize_gate_config,
)
from ci import pytest_plugin as pp


def _report(weighted=0.85, fc=0.82, latency=2.0, error=False):
    test_entry = {"summary": {"overall_score": fc}}
    if error:
        test_entry = {"error": "boom"}
    return {
        "models": {
            "gpt-4o": {
                "overall_metrics": {"weighted_score": weighted, "latency_p95": latency},
                "tests": {
                    "function_calling": test_entry,
                    "turkish_grammar": {"summary": {"overall_score": 0.90}},
                },
            }
        }
    }


def _config(**overrides):
    base = {
        "weighted_score_min": 0.70,
        "max_latency_p95": 10.0,
        "fail_on_test_error": True,
        "tests": {"function_calling": 0.80},
        "regression": {"max_weighted_drop": 0.03, "max_test_drop": 0.05},
    }
    base.update(overrides)
    return base


def test_passing_report_passes_gate():
    gate = evaluate_gate(_report(), _config())
    assert gate.passed is True
    assert gate.passed_models == ["gpt-4o"]
    assert gate.failed_models == []


def test_weighted_score_below_min_fails():
    gate = evaluate_gate(_report(weighted=0.50), _config())
    assert gate.passed is False
    assert "gpt-4o" in gate.failed_models
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "weighted_score" in failed


def test_per_test_threshold_fails():
    gate = evaluate_gate(_report(fc=0.40), _config())
    assert gate.passed is False
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "test:function_calling" in failed


def test_latency_max_fails():
    gate = evaluate_gate(_report(latency=99.0), _config())
    assert gate.passed is False
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "latency_p95" in failed


def test_test_error_fails_when_configured():
    gate = evaluate_gate(_report(error=True), _config())
    assert gate.passed is False
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "test_errors" in failed


def test_test_error_ignored_when_disabled():
    gate = evaluate_gate(_report(error=True), _config(fail_on_test_error=False))
    # function_calling threshold can't be evaluated on an errored test, so it is
    # skipped; remaining checks pass.
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "test_errors" not in failed


def test_regression_weighted_drop_fails():
    baseline = _report(weighted=0.90)
    current = _report(weighted=0.85)  # 0.05 drop > 0.03 allowed
    gate = evaluate_gate(current, _config(), baseline=baseline)
    failed = [c.name for m in gate.models for c in m.failed_checks]
    assert "regression:weighted_score" in failed


def test_regression_within_tolerance_passes():
    baseline = _report(weighted=0.87)
    current = _report(weighted=0.85)  # 0.02 drop <= 0.03 allowed
    gate = evaluate_gate(current, _config(), baseline=baseline)
    reg = [c for m in gate.models for c in m.checks if c.name == "regression:weighted_score"]
    assert reg and reg[0].passed is True
    assert reg[0].delta == pytest.approx(-0.02, abs=1e-6)


def test_empty_report_does_not_pass():
    gate = evaluate_gate({"models": {}}, _config())
    assert gate.passed is False


def test_renderers_emit_expected_shapes():
    gate = evaluate_gate(_report(weighted=0.50), _config())
    md = render_markdown(gate)
    assert "CI Gate" in md and "gpt-4o" in md and "❌" in md

    badge = json.loads(render_badge(gate))
    assert badge["schemaVersion"] == 1
    assert badge["label"] == "llm eval"
    assert badge["color"] == "red"

    js = json.loads(render_json(gate))
    assert js["passed"] is False
    assert "gpt-4o" in js["failed_models"]


def test_badge_green_when_passing():
    gate = evaluate_gate(_report(), _config())
    badge = json.loads(render_badge(gate))
    assert badge["message"] == "passing"
    assert badge["color"] == "brightgreen"


def test_normalize_fills_defaults():
    cfg = _normalize_gate_config({})
    assert cfg["fail_on_test_error"] is True
    assert cfg["tests"] == {}
    assert cfg["regression"] == {}


# --------------------------------------------------------------------------- #
# pytest_plugin helpers
# --------------------------------------------------------------------------- #

def test_assert_weighted_score_helper():
    report = _report(weighted=0.85)
    pp.assert_weighted_score(report, "gpt-4o", min_score=0.80)
    with pytest.raises(AssertionError):
        pp.assert_weighted_score(report, "gpt-4o", min_score=0.90)


def test_assert_test_score_helper():
    report = _report(fc=0.82)
    pp.assert_test_score(report, "gpt-4o", "function_calling", min_score=0.80)
    with pytest.raises(AssertionError):
        pp.assert_test_score(report, "gpt-4o", "function_calling", min_score=0.90)


def test_assert_no_regression_helper():
    baseline = _report(weighted=0.87)
    pp.assert_no_regression(_report(weighted=0.85), baseline, "gpt-4o", max_drop=0.03)
    with pytest.raises(AssertionError):
        pp.assert_no_regression(_report(weighted=0.80), baseline, "gpt-4o", max_drop=0.03)


def test_assert_gate_helper():
    pp_config = "config/ci_gate.yaml"
    # synthetic report uses tests not all present in config; should still pass
    pp.assert_gate(_report(weighted=0.85), config_path=pp_config)
    with pytest.raises(AssertionError):
        pp.assert_gate(_report(weighted=0.40), config_path=pp_config)
