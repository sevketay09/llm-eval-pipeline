"""Contract tests for experiments/differ.py."""
from __future__ import annotations
import pytest
from experiments.store import VariantResult
from experiments.differ import compute_diff, CaseDiff, _IMPROVEMENT_THRESHOLD, _REGRESSION_THRESHOLD


def _res(case_id: str, score: float, label: str = "v1", output: str = "") -> VariantResult:
    return VariantResult(variant_label=label, case_id=case_id, output=output, score=score, latency_ms=10.0)


class TestComputeDiff:
    def test_improved(self):
        base = [_res("c1", 0.4, "v1")]
        compare = [_res("c1", 0.9, "v2")]
        diffs = compute_diff(base, compare)
        assert diffs[0].verdict == "improved"
        assert diffs[0].delta > _IMPROVEMENT_THRESHOLD

    def test_regressed(self):
        base = [_res("c1", 0.9, "v1")]
        compare = [_res("c1", 0.3, "v2")]
        diffs = compute_diff(base, compare)
        assert diffs[0].verdict == "regressed"
        assert diffs[0].delta < -_REGRESSION_THRESHOLD

    def test_stable(self):
        base = [_res("c1", 0.8, "v1")]
        compare = [_res("c1", 0.82, "v2")]
        diffs = compute_diff(base, compare)
        assert diffs[0].verdict == "stable"

    def test_missing_case(self):
        base = [_res("c1", 0.8, "v1")]
        compare = []
        diffs = compute_diff(base, compare)
        assert diffs[0].verdict == "missing"

    def test_sorted_by_delta(self):
        base = [_res("c1", 0.5, "v1"), _res("c2", 0.5, "v1"), _res("c3", 0.5, "v1")]
        compare = [_res("c1", 0.9, "v2"), _res("c2", 0.5, "v2"), _res("c3", 0.1, "v2")]
        diffs = compute_diff(base, compare)
        deltas = [d.delta for d in diffs]
        assert deltas == sorted(deltas)

    def test_delta_calculation(self):
        base = [_res("c1", 0.6, "v1")]
        compare = [_res("c1", 0.8, "v2")]
        diffs = compute_diff(base, compare)
        assert abs(diffs[0].delta - 0.2) < 0.001

    def test_custom_labels(self):
        base = [_res("c1", 0.5)]
        compare = [_res("c1", 0.7)]
        diffs = compute_diff(base, compare, base_label="baseline", compare_label="experiment")
        assert diffs[0].base_label == "baseline"
        assert diffs[0].compare_label == "experiment"

    def test_to_dict(self):
        d = CaseDiff(
            case_id="c1", base_label="v1", compare_label="v2",
            base_score=0.5, compare_score=0.8,
            base_output="old", compare_output="new",
            delta=0.3, verdict="improved",
        )
        dd = d.to_dict()
        assert dd["verdict"] == "improved"
        assert dd["delta"] == 0.3
