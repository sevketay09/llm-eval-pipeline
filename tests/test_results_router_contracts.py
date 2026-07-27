import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_results_router_module():
    module_path = Path(__file__).resolve().parent.parent / "api" / "routers" / "results.py"
    spec = importlib.util.spec_from_file_location("isolated_results_router", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeReportService:
    def __init__(self):
        self.last_limit = None

    def list_reports(self, limit):
        self.last_limit = limit
        return [
            {
                "filename": "baseline.json",
                "path": "/tmp/baseline.json",
                "modified": datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
                "size_kb": 12,
                "model_count": 1,
                "suite": "smoke",
                "export_links": {
                    "raw": "/api/results/reports/baseline.json/raw",
                    "markdown": "/api/results/reports/baseline.json/markdown",
                    "html": "/api/results/reports/baseline.json/html",
                },
            }
        ]

    def get_report(self, filename):
        if filename == "missing.json":
            return None
        return {
            "filename": filename,
            "metadata": {"run_id": "report-1", "test_suite": "smoke"},
            "models": {"demo": {"tests": {}}},
            "model_scores": {"demo": 0.8},
            "model_comparison": {"demo": {"overall_score": 0.8}},
            "trends": {},
            "continuity": {
                "by_model": [
                    {
                        "model": "demo",
                        "intent_resolution": 0.8,
                        "unresolved_turn_rate": 0.2,
                        "unresolved_turns": 1,
                        "unresolved_intent_total": 2,
                    }
                ],
                "best_intent_resolution_model": "demo",
                "highest_unresolved_rate_model": "demo",
            },
            "efficiency": {},
            "disagreement": {},
            "policy": {},
            "policy_audit": {},
        }

    def get_report_raw(self, filename):
        if filename == "missing.json":
            return None
        return {"filename": filename, "raw": True}

    def get_report_markdown(self, filename):
        if filename == "missing.json":
            return None
        return "# Baseline Report\n"

    def get_report_html(self, filename):
        if filename == "missing.json":
            return None
        return "<html><body>Baseline Report</body></html>"

    def compare_reports(self, filenames):
        return {
            "baseline.json": {
                "model_scores": {"demo": 0.8},
                "continuity": {
                    "by_model": [
                        {
                            "model": "demo",
                            "intent_resolution": 0.81,
                            "unresolved_turn_rate": 0.2,
                            "unresolved_turns": 1,
                            "unresolved_intent_total": 2,
                        }
                    ],
                    "best_intent_resolution_model": "demo",
                    "highest_unresolved_rate_model": "demo",
                },
                "model_comparison": {
                    "demo": {
                        "overall_score": 0.8,
                    }
                },
                "prompt_version": "judge-v1",
                "schema_version": "2.0",
                "metric_version": "metric-v1",
            },
            "candidate.json": {
                "model_scores": {"demo": 0.9},
                "continuity": {
                    "by_model": [
                        {
                            "model": "demo",
                            "intent_resolution": 0.9,
                            "unresolved_turn_rate": 0.1,
                            "unresolved_turns": 0,
                            "unresolved_intent_total": 0,
                        }
                    ],
                    "best_intent_resolution_model": "demo",
                    "highest_unresolved_rate_model": "demo",
                },
                "model_comparison": {
                    "demo": {
                        "overall_score": 0.9,
                    }
                },
                "prompt_version": "judge-v2",
                "schema_version": "2.1",
                "metric_version": "metric-v2",
            },
        }


def _build_client(service, rag_eval_service=None):
    results_router_module = _load_results_router_module()
    app = FastAPI()
    app.include_router(results_router_module.router, prefix="/api")
    app.dependency_overrides[results_router_module.get_report_service] = lambda: service
    if rag_eval_service is not None:
        app.dependency_overrides[results_router_module.get_rag_eval_service] = lambda: rag_eval_service
    return TestClient(app)


class _RagFakeReportService(_FakeReportService):
    """get_report_raw() returns a report shaped like a real eval_*.json file,
    with one RAG-eligible case (has a "contexts" field) and one plain
    (non-RAG) case that evaluate_rag_report must skip."""

    def get_report_raw(self, filename):
        if filename == "missing.json":
            return None
        return {
            "models": {
                "demo-model": {
                    "tests": {
                        "rag_test": {
                            "results": [
                                {
                                    "question": "What is the capital of France?",
                                    "contexts": ["Paris is the capital of France."],
                                    "answer": "Paris.",
                                },
                            ]
                        },
                        "turkish_grammar": {
                            "results": [
                                {"question": "not a rag case", "answer": "no contexts field here"},
                            ]
                        },
                    }
                }
            }
        }


class ResultsRouterContractTests(unittest.TestCase):
    def test_list_reports_returns_typed_items(self):
        service = _FakeReportService()
        client = _build_client(service)

        response = client.get("/api/results/reports?limit=5")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(service.last_limit, 5)
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["filename"], "baseline.json")
        self.assertEqual(body[0]["suite"], "smoke")
        self.assertEqual(body[0]["export_links"]["html"], "/api/results/reports/baseline.json/html")

    def test_get_report_returns_typed_summary_and_404_when_missing(self):
        client = _build_client(_FakeReportService())

        response = client.get("/api/results/reports/baseline.json")
        missing = client.get("/api/results/reports/missing.json")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["filename"], "baseline.json")
        self.assertEqual(body["model_scores"]["demo"], 0.8)
        self.assertEqual(body["continuity"]["best_intent_resolution_model"], "demo")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"], "Report 'missing.json' not found")

    def test_raw_markdown_and_html_endpoints_preserve_content_and_404_behavior(self):
        client = _build_client(_FakeReportService())

        raw_response = client.get("/api/results/reports/baseline.json/raw")
        markdown_response = client.get("/api/results/reports/baseline.json/markdown")
        html_response = client.get("/api/results/reports/baseline.json/html")
        missing_raw = client.get("/api/results/reports/missing.json/raw")
        missing_markdown = client.get("/api/results/reports/missing.json/markdown")
        missing_html = client.get("/api/results/reports/missing.json/html")

        self.assertEqual(raw_response.status_code, 200)
        self.assertEqual(raw_response.json(), {"filename": "baseline.json", "raw": True})

        self.assertEqual(markdown_response.status_code, 200)
        self.assertEqual(markdown_response.text, "# Baseline Report\n")
        self.assertIn("text/plain", markdown_response.headers["content-type"])

        self.assertEqual(html_response.status_code, 200)
        self.assertEqual(html_response.text, "<html><body>Baseline Report</body></html>")
        self.assertIn("text/html", html_response.headers["content-type"])

        self.assertEqual(missing_raw.status_code, 404)
        self.assertEqual(missing_markdown.status_code, 404)
        self.assertEqual(missing_html.status_code, 404)
        self.assertEqual(missing_raw.json()["detail"], "Report 'missing.json' not found")
        self.assertEqual(missing_markdown.json()["detail"], "Report 'missing.json' not found")
        self.assertEqual(missing_html.json()["detail"], "Report 'missing.json' not found")

    def test_compare_endpoint_preserves_compare_contract_fields(self):
        client = _build_client(_FakeReportService())

        response = client.post("/api/results/compare", json=["baseline.json", "candidate.json"])

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["baseline.json"]["prompt_version"], "judge-v1")
        self.assertEqual(body["baseline.json"]["schema_version"], "2.0")
        self.assertEqual(body["baseline.json"]["metric_version"], "metric-v1")
        self.assertEqual(body["baseline.json"]["model_comparison"]["demo"]["overall_score"], 0.8)
        self.assertEqual(body["baseline.json"]["continuity"]["best_intent_resolution_model"], "demo")
        self.assertEqual(body["candidate.json"]["schema_version"], "2.1")
        self.assertEqual(body["candidate.json"]["metric_version"], "metric-v2")

    def test_compare_endpoint_requires_at_least_two_reports(self):
        client = _build_client(_FakeReportService())

        response = client.post("/api/results/compare", json=["baseline.json"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Need at least 2 reports to compare")


class ResultsRagEvalEndpointTests(unittest.TestCase):
    def test_aggregates_rag_cases_and_skips_non_rag_results(self):
        from api.services.rag_eval_service import RagEvalService

        client = _build_client(_RagFakeReportService(), rag_eval_service=RagEvalService())

        response = client.get("/api/results/reports/baseline.json/rag-eval")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_rag_cases"], 1)  # the turkish_grammar case has no "contexts" field
        self.assertIn("demo-model", body["models"])
        self.assertEqual(body["models"]["demo-model"]["rag_case_count"], 1)
        self.assertGreater(body["models"]["demo-model"]["avg_faithfulness"], 0.0)

    def test_missing_report_returns_404(self):
        from api.services.rag_eval_service import RagEvalService

        client = _build_client(_RagFakeReportService(), rag_eval_service=RagEvalService())

        response = client.get("/api/results/reports/missing.json/rag-eval")

        self.assertEqual(response.status_code, 404)

    def test_unknown_embedding_model_returns_404(self):
        from api.services.rag_eval_service import RagEvalService

        def failing_factory(model_key, config_path):
            raise ValueError(f"Embedding model '{model_key}' not found in config")

        client = _build_client(
            _RagFakeReportService(),
            rag_eval_service=RagEvalService(embedding_adapter_factory=failing_factory),
        )

        response = client.get("/api/results/reports/baseline.json/rag-eval?embedding_model=does-not-exist")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()