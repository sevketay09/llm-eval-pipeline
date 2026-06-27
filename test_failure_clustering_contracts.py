"""Contract tests for failure_clustering module.

Deterministic, offline-only tests. No real embeddings or LLM calls.
Uses seeded random and injectable functions for all external dependencies.

Test contract:
- extract_failures: 4 tests
- cluster_failures: 4 tests
- label_clusters: 2 tests
- compute_failure_summary: 3+ tests
Total: 13+ contract tests, all deterministic.
"""
import unittest
from typing import Any

import numpy as np

from analysis.failure_clustering import (
    extract_failures,
    cluster_failures,
    label_clusters,
    compute_failure_summary,
)


# ============================================================================= #
# ExtractFailuresContractTests
# ============================================================================= #

class ExtractFailuresContractTests(unittest.TestCase):
    """Contract tests for extract_failures()"""

    def test_extracts_cases_below_threshold(self) -> None:
        """Cases with score < threshold are extracted."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.5},
                                    "question": "Q1",
                                }
                            ]
                        }
                    }
                }
            }
        }
        failures = extract_failures(report, threshold=0.6)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["case_id"], "c1")
        self.assertEqual(failures[0]["score"], 0.5)
        self.assertEqual(failures[0]["model"], "model-a")
        self.assertEqual(failures[0]["test"], "test-1")

    def test_skips_cases_above_or_equal_threshold(self) -> None:
        """Cases with score >= threshold are not extracted."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.8},
                                    "question": "Q1",
                                },
                                {
                                    "case_id": "c2",
                                    "scores": {"overall_score": 0.6},
                                    "question": "Q2",
                                }
                            ]
                        }
                    }
                }
            }
        }
        failures = extract_failures(report, threshold=0.6)
        # Only c1 is skipped; c2 (score=0.6 exactly) is not extracted
        self.assertEqual(len(failures), 0)

    def test_error_cases_always_extracted(self) -> None:
        """Cases with 'error' key are always extracted with score=0.0."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.95},
                                    "question": "Q1",
                                    "error": "timeout",
                                }
                            ]
                        }
                    }
                }
            }
        }
        failures = extract_failures(report, threshold=0.6)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["score"], 0.0)
        self.assertEqual(failures[0]["error"], "timeout")

    def test_text_fallback_chain(self) -> None:
        """Text field uses fallback chain: question→input_text→prompt→case_id."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.5},
                                    "input_text": "input",
                                    "prompt": "prompt",
                                },
                                {
                                    "case_id": "c2",
                                    "scores": {"overall_score": 0.5},
                                    "question": "question",
                                    "input_text": "input",
                                },
                                {
                                    "case_id": "c3",
                                    "scores": {"overall_score": 0.5},
                                },
                            ]
                        }
                    }
                }
            }
        }
        failures = extract_failures(report)

        # c3 should use case_id (no question, input_text, or prompt)
        self.assertEqual(failures[2]["text"], "c3")
        # c2 should prefer question over input_text
        self.assertEqual(failures[1]["text"], "question")
        # c1 should use input_text (no question)
        self.assertEqual(failures[0]["text"], "input")


# ============================================================================= #
# ClusterFailuresContractTests
# ============================================================================= #

class ClusterFailuresContractTests(unittest.TestCase):
    """Contract tests for cluster_failures()"""

    def test_fewer_than_2_failures_returns_empty(self) -> None:
        """0 or 1 failure returns empty list."""
        self.assertEqual(cluster_failures([]), [])
        self.assertEqual(
            cluster_failures([{"case_id": "c1", "text": "t1", "score": 0.5}]),
            []
        )

    def test_clusters_with_fake_embed_fn(self) -> None:
        """Clusters are created with injectable embed_fn."""
        failures = [
            {"case_id": f"c{i}", "text": f"text {i}", "score": 0.5, "model": "m1"}
            for i in range(10)
        ]

        # Fake embed_fn: seeded random vectors
        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 5)

        clusters = cluster_failures(failures, n_clusters=3, embed_fn=fake_embed)
        self.assertEqual(len(clusters), 3)

        for cluster in clusters:
            self.assertIn("cluster_id", cluster)
            self.assertIn("size", cluster)
            self.assertIn("members", cluster)
            self.assertGreater(cluster["size"], 0)

    def test_auto_n_clusters_selection(self) -> None:
        """Auto n_clusters selection: min(max(2, len//3), 8)."""
        failures = [
            {"case_id": f"c{i}", "text": f"text {i}", "score": 0.5, "model": "m1"}
            for i in range(10)
        ]

        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 5)

        clusters = cluster_failures(failures, n_clusters=None, embed_fn=fake_embed)
        # 10 failures → n_clusters = min(max(2, 10//3), 8) = min(3, 8) = 3
        self.assertEqual(len(clusters), 3)

    def test_cluster_output_fields(self) -> None:
        """Each cluster has all required fields."""
        failures = [
            {"case_id": f"c{i}", "text": f"text {i}", "score": 0.5, "model": "m1"}
            for i in range(5)
        ]

        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 2)

        clusters = cluster_failures(failures, n_clusters=2, embed_fn=fake_embed)

        for cluster in clusters:
            self.assertIn("cluster_id", cluster)
            self.assertIn("size", cluster)
            self.assertIn("label", cluster)
            self.assertIn("members", cluster)
            self.assertIn("centroid_text", cluster)
            self.assertIn("avg_score", cluster)

            self.assertIsInstance(cluster["cluster_id"], int)
            self.assertIsInstance(cluster["size"], int)
            self.assertIsInstance(cluster["label"], str)
            self.assertIsInstance(cluster["members"], list)
            self.assertIsInstance(cluster["centroid_text"], str)
            self.assertIsInstance(cluster["avg_score"], float)


# ============================================================================= #
# LabelClustersContractTests
# ============================================================================= #

class LabelClustersContractTests(unittest.TestCase):
    """Contract tests for label_clusters()"""

    def test_default_keyword_label_uses_most_common_words(self) -> None:
        """Default label comes from most common words in cluster texts."""
        clusters = [
            {
                "cluster_id": 0,
                "members": [
                    {"text": "numerical reasoning problem"},
                    {"text": "numerical calculation error"},
                    {"text": "math reasoning issue"},
                ],
                "size": 3,
            }
        ]

        labeled = label_clusters(clusters)
        label = labeled[0]["label"]

        # Should be non-empty and generated from texts
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)
        # Should contain words from the texts (excluding stopwords)
        self.assertTrue(
            any(word in label.lower() for word in ["numerical", "reasoning", "math"])
        )

    def test_custom_label_fn_called(self) -> None:
        """Custom label_fn is called and its result is used."""
        clusters = [
            {
                "cluster_id": 0,
                "members": [
                    {"text": "text1"},
                    {"text": "text2"},
                ],
                "size": 2,
            }
        ]

        call_count = [0]

        def custom_label_fn(texts: list[str]) -> str:
            call_count[0] += 1
            return "custom_label"

        labeled = label_clusters(clusters, label_fn=custom_label_fn)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(labeled[0]["label"], "custom_label")


# ============================================================================= #
# ComputeFailureSummaryContractTests
# ============================================================================= #

class ComputeFailureSummaryContractTests(unittest.TestCase):
    """Contract tests for compute_failure_summary()"""

    def test_no_failures_returns_empty_result(self) -> None:
        """When no failures, returns proper empty structure."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.9},
                                    "question": "Q1",
                                }
                            ]
                        }
                    }
                }
            }
        }

        summary = compute_failure_summary(report, threshold=0.6)

        self.assertEqual(summary["total_failures"], 0)
        self.assertEqual(summary["threshold"], 0.6)
        self.assertEqual(summary["clusters"], [])
        self.assertEqual(summary["model_breakdown"], {})
        self.assertEqual(summary["category_breakdown"], {})

    def test_summary_fields_present(self) -> None:
        """Summary has all required top-level fields."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.5},
                                    "question": "Q1",
                                    "category": "math",
                                }
                            ]
                        }
                    }
                }
            }
        }

        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 2)

        summary = compute_failure_summary(report, embed_fn=fake_embed)

        self.assertIn("total_failures", summary)
        self.assertIn("threshold", summary)
        self.assertIn("clusters", summary)
        self.assertIn("model_breakdown", summary)
        self.assertIn("category_breakdown", summary)

    def test_model_and_category_breakdown_counts(self) -> None:
        """Model and category breakdowns have correct counts."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c1",
                                    "scores": {"overall_score": 0.5},
                                    "question": "Q1",
                                    "category": "math",
                                },
                                {
                                    "case_id": "c2",
                                    "scores": {"overall_score": 0.5},
                                    "question": "Q2",
                                    "category": "lang",
                                },
                            ]
                        }
                    }
                },
                "model-b": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": "c3",
                                    "scores": {"overall_score": 0.4},
                                    "question": "Q3",
                                    "category": "math",
                                }
                            ]
                        }
                    }
                }
            }
        }

        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 2)

        summary = compute_failure_summary(report, embed_fn=fake_embed)

        self.assertEqual(summary["total_failures"], 3)
        self.assertEqual(summary["model_breakdown"]["model-a"], 2)
        self.assertEqual(summary["model_breakdown"]["model-b"], 1)
        self.assertEqual(summary["category_breakdown"]["math"], 2)
        self.assertEqual(summary["category_breakdown"]["lang"], 1)


# ============================================================================= #
# Additional integration contract tests
# ============================================================================= #

class FailureClusteringIntegrationContractTests(unittest.TestCase):
    """Integration contract tests across multiple functions."""

    def test_end_to_end_pipeline_with_real_report_structure(self) -> None:
        """Full pipeline: extract → cluster → label → summary."""
        report = {
            "models": {
                "gpt-4o": {
                    "tests": {
                        "math_test": {
                            "results": [
                                {
                                    "case_id": "m1",
                                    "scores": {"overall_score": 0.3},
                                    "question": "5 plus 3",
                                    "category": "arithmetic",
                                },
                                {
                                    "case_id": "m2",
                                    "scores": {"overall_score": 0.2},
                                    "question": "10 times 5",
                                    "category": "arithmetic",
                                },
                                {
                                    "case_id": "m3",
                                    "scores": {"overall_score": 0.9},
                                    "question": "easy question",
                                    "category": "simple",
                                }
                            ]
                        }
                    }
                },
                "gpt-3.5": {
                    "tests": {
                        "math_test": {
                            "results": [
                                {
                                    "case_id": "m4",
                                    "scores": {"overall_score": 0.4},
                                    "question": "2 divided by 2",
                                    "category": "arithmetic",
                                    "error": "timeout",
                                }
                            ]
                        }
                    }
                }
            }
        }

        def fake_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 3)

        summary = compute_failure_summary(
            report, threshold=0.6, n_clusters=2, embed_fn=fake_embed
        )

        # Should have extracted 3 failures (m1, m2, m4; m3 >= 0.6)
        self.assertEqual(summary["total_failures"], 3)
        self.assertEqual(len(summary["clusters"]), 2)

        # Model breakdown
        self.assertEqual(summary["model_breakdown"]["gpt-4o"], 2)
        self.assertEqual(summary["model_breakdown"]["gpt-3.5"], 1)

        # Category breakdown
        self.assertEqual(summary["category_breakdown"]["arithmetic"], 3)

        # All clusters should be labeled
        for cluster in summary["clusters"]:
            self.assertGreater(len(cluster["label"]), 0)

    def test_deterministic_clustering_same_seed(self) -> None:
        """Same report and seed produce identical clusters."""
        report = {
            "models": {
                "model-a": {
                    "tests": {
                        "test-1": {
                            "results": [
                                {
                                    "case_id": f"c{i}",
                                    "scores": {"overall_score": 0.3},
                                    "question": f"question {i}",
                                    "category": "test",
                                }
                                for i in range(6)
                            ]
                        }
                    }
                }
            }
        }

        def fake_embed_1(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 3)

        def fake_embed_2(texts: list[str]) -> np.ndarray:
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 3)

        summary1 = compute_failure_summary(
            report, threshold=0.6, n_clusters=2, embed_fn=fake_embed_1
        )
        summary2 = compute_failure_summary(
            report, threshold=0.6, n_clusters=2, embed_fn=fake_embed_2
        )

        # Cluster sizes should be identical
        sizes1 = sorted([c["size"] for c in summary1["clusters"]])
        sizes2 = sorted([c["size"] for c in summary2["clusters"]])
        self.assertEqual(sizes1, sizes2)


if __name__ == "__main__":
    unittest.main()
