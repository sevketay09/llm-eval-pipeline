"""Contract tests for analysis/significance.py.

Synthetic reports exercise CI computation, paired pairwise significance,
small-sample warnings, edge cases (errored/missing tests, identical scores),
determinism, and renderers — no real evaluation needed.
"""
import json

import pytest

from analysis.significance import (
    extract_test_scores,
    per_model_intervals,
    pairwise_comparisons,
    compute_significance,
    render_markdown,
    render_json,
    render_text,
    SMALL_SAMPLE_N,
)


def _report(model_scores, weighted=None, errored=None):
    """model_scores: {model: {test: score}}. errored: {model: [test,...]}."""
    errored = errored or {}
    weighted = weighted or {}
    models = {}
    for model, tests in model_scores.items():
        test_entries = {
            t: {"summary": {"overall_score": s}} for t, s in tests.items()
        }
        for t in errored.get(model, []):
            test_entries[t] = {"error": "boom"}
        models[model] = {
            "overall_metrics": {"weighted_score": weighted.get(model, sum(tests.values()) / max(len(tests), 1))},
            "tests": test_entries,
        }
    return {"models": models}


def _big(model, offset=0.0, n=12):
    return {f"t{i}": min(1.0, 0.6 + 0.01 * i + offset) for i in range(n)}


def test_extract_skips_errored_and_nonnumeric():
    rep = _report(
        {"m1": {"a": 0.8, "b": 0.6}},
        errored={"m1": ["c"]},
    )
    # inject a non-numeric score
    rep["models"]["m1"]["tests"]["d"] = {"summary": {"overall_score": "n/a"}}
    scores = extract_test_scores(rep)
    assert scores["m1"] == {"a": 0.8, "b": 0.6}


def test_per_model_intervals_basic():
    rep = _report({"m1": _big("m1")})
    intervals = per_model_intervals(rep)
    m = intervals["m1"]
    assert m["n_tests"] == 12
    assert m["ci_lower"] <= m["mean_test_score"] <= m["ci_upper"]
    assert m["small_sample"] is False


def test_small_sample_flagged():
    rep = _report({"m1": {"a": 0.8, "b": 0.7}})
    intervals = per_model_intervals(rep)
    assert intervals["m1"]["small_sample"] is True
    assert intervals["m1"]["n_tests"] < SMALL_SAMPLE_N


def test_pairwise_significant_difference():
    # m1 consistently ~0.1 higher across 12 shared tests -> significant
    rep = _report({"m1": _big("m1", offset=0.1), "m2": _big("m2", offset=0.0)})
    pairs = pairwise_comparisons(rep)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["n_shared_tests"] == 12
    assert p["is_significant"] is True
    assert p["winner"] == "m1"
    assert p["mean_difference"] > 0


def test_pairwise_no_difference_identical():
    rep = _report({"m1": _big("m1"), "m2": _big("m1")})
    p = pairwise_comparisons(rep)[0]
    assert p["is_significant"] is False
    assert p["winner"] is None


def test_pairwise_insufficient_shared():
    rep = _report({"m1": {"a": 0.8}, "m2": {"b": 0.7}})  # no shared tests
    p = pairwise_comparisons(rep)[0]
    assert p["n_shared_tests"] == 0
    assert p["is_significant"] is False
    assert "insufficient" in p["verdict"]


def test_determinism():
    rep = _report({"m1": _big("m1", offset=0.05), "m2": _big("m2")})
    a = compute_significance(rep)
    b = compute_significance(rep)
    assert render_json(a) == render_json(b)


def test_compute_significance_warnings_on_small_sample():
    rep = _report({"m1": {"a": 0.8, "b": 0.7}, "m2": {"a": 0.6, "b": 0.5}})
    result = compute_significance(rep)
    assert result["warnings"]  # small sample warnings present


def test_renderers_smoke():
    rep = _report({"m1": _big("m1", offset=0.1), "m2": _big("m2")})
    result = compute_significance(rep)

    md = render_markdown(result)
    assert "İstatistiksel" in md and "m1" in md and "m2" in md

    txt = render_text(result)
    assert "STATISTICAL COMPARISON" in txt

    js = json.loads(render_json(result))
    assert "per_model" in js and "pairwise" in js
    assert js["alpha"] == 0.05


def test_three_models_yield_three_pairs():
    rep = _report({"m1": _big("m1"), "m2": _big("m2"), "m3": _big("m3")})
    pairs = pairwise_comparisons(rep)
    assert len(pairs) == 3
