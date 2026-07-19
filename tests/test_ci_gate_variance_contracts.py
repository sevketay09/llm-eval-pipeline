"""Contract tests: variance-aware CI gate (bootstrap CI vs bare point estimate)."""
import unittest

from ci.gate import evaluate_gate, _bootstrap_ci, _test_case_scores


def make_report(scores, overall):
    return {
        "models": {
            "m1": {
                "overall_metrics": {"weighted_score": 0.9},
                "tests": {
                    "qa_test": {
                        "summary": {"overall_score": overall},
                        "results": [{"scores": {"judge_score": s}} for s in scores],
                    }
                },
            }
        }
    }


def gate_config(bound="upper", enabled=True):
    return {
        "tests": {"qa_test": 0.75},
        "fail_on_test_error": False,
        "variance_aware": {"enabled": enabled, "confidence": 0.95, "n_bootstrap": 500, "bound": bound},
    }


def qa_check(gate_report):
    return next(c for c in gate_report.models[0].checks if c.name == "test:qa_test")


class BootstrapCiTests(unittest.TestCase):
    def test_deterministic(self):
        scores = [0.2, 0.5, 0.8, 1.0, 0.4]
        self.assertEqual(_bootstrap_ci(scores), _bootstrap_ci(scores))

    def test_ci_contains_mean_for_homogeneous_data(self):
        lo, hi = _bootstrap_ci([0.8] * 10)
        self.assertAlmostEqual(lo, 0.8)
        self.assertAlmostEqual(hi, 0.8)

    def test_case_score_extraction(self):
        td = {"results": [{"scores": {"judge_score": 1.0}}, {"scores": {"judge_score": "x"}}, {}]}
        self.assertEqual(_test_case_scores(td), [1.0])


class VarianceAwareGateTests(unittest.TestCase):
    def test_disabled_keeps_point_estimate_behavior(self):
        report = make_report([0.5, 1.0, 0.5, 1.0, 0.5, 1.0], overall=0.70)
        gate = evaluate_gate(report, gate_config(enabled=False))
        check = qa_check(gate)
        self.assertFalse(check.passed)  # 0.70 < 0.75
        self.assertIsNone(check.ci_lower)

    def test_upper_bound_rescues_noisy_small_sample(self):
        # Mean 0.70 < 0.75 threshold, but with 6 noisy cases the CI upper
        # edge crosses the threshold → inconclusive, don't break the build.
        report = make_report([0.5, 1.0, 0.5, 1.0, 0.5, 0.7], overall=0.70)
        gate = evaluate_gate(report, gate_config(bound="upper"))
        check = qa_check(gate)
        self.assertTrue(check.passed)
        self.assertGreaterEqual(check.ci_upper, 0.75)
        self.assertEqual(check.sample_size, 6)
        self.assertIn("CI95%", check.detail)

    def test_upper_bound_still_fails_confidently_bad_model(self):
        report = make_report([0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0], overall=0.0625)
        gate = evaluate_gate(report, gate_config(bound="upper"))
        self.assertFalse(qa_check(gate).passed)

    def test_lower_bound_is_strict(self):
        # Mean 0.83 above threshold, but noisy → pessimistic edge below 0.75 → fail
        report = make_report([1.0, 1.0, 0.5, 1.0, 0.5, 1.0], overall=0.8333)
        gate = evaluate_gate(report, gate_config(bound="lower"))
        check = qa_check(gate)
        self.assertFalse(check.passed)
        self.assertLess(check.ci_lower, 0.75)

    def test_no_case_scores_falls_back_to_point_estimate(self):
        report = make_report([], overall=0.9)
        report["models"]["m1"]["tests"]["qa_test"]["results"] = []
        gate = evaluate_gate(report, gate_config())
        check = qa_check(gate)
        self.assertTrue(check.passed)
        self.assertIsNone(check.ci_lower)

    def test_check_serializes_with_ci_fields(self):
        report = make_report([0.5, 1.0, 0.5, 1.0, 0.5, 0.7], overall=0.70)
        gate = evaluate_gate(report, gate_config())
        payload = gate.to_dict()
        check = next(c for c in payload["models"][0]["checks"] if c["name"] == "test:qa_test")
        self.assertIn("ci_upper", check)
        self.assertIn("sample_size", check)


if __name__ == "__main__":
    unittest.main()
