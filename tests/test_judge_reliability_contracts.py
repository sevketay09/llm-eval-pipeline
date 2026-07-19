"""Contract tests for analysis/judge_reliability.py (kappa, spearman, reliability verdict)."""
import unittest

from analysis.judge_reliability import (
    spearman_rho,
    cohens_kappa,
    score_bucket,
    compute_judge_reliability,
)


class SpearmanTests(unittest.TestCase):
    def test_perfect_monotonic(self):
        self.assertAlmostEqual(spearman_rho([0.1, 0.5, 0.9], [0.2, 0.6, 1.0]), 1.0)

    def test_perfect_inverse(self):
        self.assertAlmostEqual(spearman_rho([0.1, 0.5, 0.9], [1.0, 0.6, 0.2]), -1.0)

    def test_constant_series_undefined(self):
        self.assertIsNone(spearman_rho([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]))

    def test_ties_average_ranks(self):
        # Ties must not crash and stay in [-1, 1]
        rho = spearman_rho([0.5, 0.5, 1.0, 0.0], [0.4, 0.6, 1.0, 0.1])
        self.assertIsNotNone(rho)
        self.assertGreaterEqual(rho, -1.0)
        self.assertLessEqual(rho, 1.0)

    def test_length_mismatch(self):
        self.assertIsNone(spearman_rho([1, 2], [1, 2, 3]))


class KappaTests(unittest.TestCase):
    def test_hand_computed_value(self):
        # observed=0.75, expected=0.5 → kappa=0.5
        a = ["low", "low", "mid", "mid"]
        b = ["low", "mid", "mid", "mid"]
        self.assertAlmostEqual(cohens_kappa(a, b), 0.5)

    def test_perfect_agreement(self):
        a = ["low", "mid", "high", "mid"]
        self.assertAlmostEqual(cohens_kappa(a, list(a)), 1.0)

    def test_both_constant_same_category_undefined(self):
        self.assertIsNone(cohens_kappa(["high"] * 4, ["high"] * 4))

    def test_score_buckets(self):
        self.assertEqual(score_bucket(0.0), "low")
        self.assertEqual(score_bucket(0.5), "mid")
        self.assertEqual(score_bucket(1.0), "high")


class ComputeReliabilityTests(unittest.TestCase):
    def test_insufficient_data(self):
        result = compute_judge_reliability([(1.0, 1.0)] * 4)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertIsNone(result["cohens_kappa"])
        self.assertEqual(result["n"], 4)

    def test_reliable_judge(self):
        pairs = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.0, 1.0), (0.0, 0.0), (0.5, 0.6)]
        result = compute_judge_reliability(pairs)
        self.assertEqual(result["verdict"], "reliable")
        self.assertGreaterEqual(result["cohens_kappa"], 0.6)
        self.assertGreater(result["spearman_rho"], 0.8)
        self.assertLessEqual(result["mean_absolute_error"], 0.15)

    def test_uncalibrated_judge(self):
        # Judge says high on everything the human scored low
        pairs = [(1.0, 0.0), (0.9, 0.1), (1.0, 0.2), (0.8, 0.0), (1.0, 0.1), (0.9, 0.0)]
        result = compute_judge_reliability(pairs)
        self.assertEqual(result["verdict"], "needs_calibration")
        self.assertGreater(result["mean_bias"], 0.5)  # systematically too generous

    def test_output_json_serializable(self):
        import json
        pairs = [(0.2, 0.3), (0.4, 0.5), (0.6, 0.4), (0.8, 0.9), (1.0, 0.8)]
        json.dumps(compute_judge_reliability(pairs))


if __name__ == "__main__":
    unittest.main()
