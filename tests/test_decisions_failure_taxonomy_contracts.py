"""Contract tests for decisions/failure_taxonomy.py and its wiring into
analysis/failure_clustering.py and the failure-clustering API. Offline
only — MockDecisionClient, no network.
"""
from __future__ import annotations

import unittest

from analysis.failure_clustering import compute_failure_summary
from decisions.failure_taxonomy import DEFAULT_FAILURE_TAXONOMY, make_classify_fn
from decisions.mock_client import MockDecisionClient


def _report(n_failing=4):
    results = [
        {"case_id": f"c{i}", "question": f"question {i} about something", "scores": {"overall_score": 0.1}}
        for i in range(n_failing)
    ]
    return {"models": {"m1": {"tests": {"t1": {"results": results}}}}}


class MakeClassifyFnContractTests(unittest.TestCase):
    def test_error_failure_skips_decision_call(self):
        client = MockDecisionClient()
        classify_fn = make_classify_fn(client)
        failures = [{"test": "t", "category": "c", "score": 0.0, "text": "x", "error": "timeout"}]
        results = classify_fn(failures)
        self.assertEqual(results, [{"label": "infra_error", "confidence": 1.0}])
        self.assertEqual(client.stats.snapshot()["requests"], 0)

    def test_classifies_non_error_failure(self):
        client = MockDecisionClient(overrides={"category": "retrieval_miss"})
        classify_fn = make_classify_fn(client)
        failures = [{"test": "t", "category": "c", "score": 0.2, "text": "missing context", "error": None}]
        results = classify_fn(failures)
        self.assertEqual(results[0]["label"], "retrieval_miss")

    def test_default_taxonomy_has_expected_labels(self):
        self.assertIn("hallucination", DEFAULT_FAILURE_TAXONOMY)
        self.assertIn("infra_error", DEFAULT_FAILURE_TAXONOMY)
        self.assertIn("other", DEFAULT_FAILURE_TAXONOMY)


class ComputeFailureSummaryTaxonomyContractTests(unittest.TestCase):
    def test_no_classify_fn_has_no_taxonomy_breakdown(self):
        summary = compute_failure_summary(_report())
        self.assertNotIn("taxonomy_breakdown", summary)

    def test_classify_fn_labels_clusters_and_breakdown(self):
        client = MockDecisionClient(overrides={"category": "retrieval_miss"})
        classify_fn = make_classify_fn(client)
        summary = compute_failure_summary(_report(n_failing=6), classify_fn=classify_fn)

        self.assertIn("taxonomy_breakdown", summary)
        self.assertEqual(summary["taxonomy_breakdown"], {"retrieval_miss": 6})
        for cluster in summary["clusters"]:
            self.assertEqual(cluster["taxonomy_label"], "retrieval_miss")
            self.assertIn("retrieval_miss (100%)", cluster["label"])
            for member in cluster["members"]:
                self.assertEqual(member["taxonomy_label"], "retrieval_miss")


class FailureClusteringServiceDecisionContractTests(unittest.TestCase):
    def test_service_requires_decision_model_for_decision_labeling(self):
        from api.services.failure_clustering_service import FailureClusteringService

        svc = FailureClusteringService()
        with self.assertRaises(ValueError):
            svc.cluster(_report(), labeling="decision")

    def test_service_uses_decision_client_factory(self):
        from api.services.failure_clustering_service import FailureClusteringService

        def factory(model_key, config_path):
            return MockDecisionClient(overrides={"category": "hallucination"})

        svc = FailureClusteringService(decision_client_factory=factory)
        response = svc.cluster(_report(), labeling="decision", decision_model="demo-jev")
        self.assertEqual(response.taxonomy_breakdown, {"hallucination": 4})


if __name__ == "__main__":
    unittest.main()
