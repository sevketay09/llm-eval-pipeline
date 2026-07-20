import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.services.skill_eval_service import SkillEvalService

SKILL_TEXT = """---
name: csv-report
description: Generates weekly CSV sales reports with totals per region.
---
# Usage
Load the CSV, group by region, write totals to report.csv.
"""

TASK = "Haftalık satış CSV'sinden bölge bazlı toplam raporu üret."

FIT_VERDICT = {
    "criteria": {
        name: {"score": 0.8, "evidence": "group by region", "reasoning": "ok"}
        for name in (
            "scope_coverage",
            "instruction_clarity",
            "completeness",
            "convention_alignment",
            "efficiency_risk",
        )
    },
    "gaps": [],
    "suggestions": ["Add delimiter option"],
}


class StubAdapter:
    def __init__(self, content):
        self.content = content

    def generate(self, messages, response_format=None, max_tokens=None):
        return {"content": self.content, "latency": 0.1, "usage": {"total_tokens": 5}}


def _stub_factory(content=None):
    def factory(model_key, config_path):
        if model_key == "missing-model":
            raise ValueError(f"Model '{model_key}' not found in config")
        return StubAdapter(content if content is not None else json.dumps(FIT_VERDICT))

    return factory


def _load_router_module():
    module_path = Path(__file__).resolve().parent.parent / "api" / "routers" / "skill_eval.py"
    spec = importlib.util.spec_from_file_location("isolated_skill_eval_router", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _client(tmpdir, content=None):
    module = _load_router_module()
    module._service = SkillEvalService(reports_dir=tmpdir, adapter_factory=_stub_factory(content))
    app = FastAPI()
    app.include_router(module.router, prefix="/api")
    return TestClient(app)


class SkillEvalServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = SkillEvalService(reports_dir=self.tmpdir, adapter_factory=_stub_factory())

    def test_lint_delegates_to_skill_lint(self):
        report = self.service.lint(SKILL_TEXT)
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["summary"]["name"], "csv-report")

    def test_full_combines_lint_and_fit_and_saves(self):
        report = self.service.full(SKILL_TEXT, TASK, "demo-model", save=True)
        self.assertEqual(report["combined_basis"], "lint+fit")
        self.assertEqual(report["combined_score"], round(0.5 * 1.0 + 0.5 * 0.8, 4))
        self.assertTrue(Path(report["report_path"]).exists())

    def test_full_falls_back_to_lint_only_when_judge_unusable(self):
        service = SkillEvalService(reports_dir=self.tmpdir, adapter_factory=_stub_factory("garbage"))
        report = service.full(SKILL_TEXT, TASK, "demo-model", save=False)
        self.assertEqual(report["combined_basis"], "lint_only")
        self.assertEqual(report["combined_score"], 1.0)
        self.assertIsNone(report["fit"])
        self.assertNotIn("report_path", report)

    def test_list_reports_returns_saved_summaries_newest_first(self):
        self.service.full(SKILL_TEXT, TASK, "demo-model", save=True)
        self.service.full(SKILL_TEXT, TASK, "demo-model", save=True)
        reports = self.service.list_reports()
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["skill_name"], "csv-report")
        self.assertEqual(reports[0]["verdict"], "fit")
        self.assertGreaterEqual(reports[0]["filename"], reports[1]["filename"])

    def test_get_report_blocks_path_traversal(self):
        saved = self.service.full(SKILL_TEXT, TASK, "demo-model", save=True)
        filename = Path(saved["report_path"]).name
        self.assertIsNotNone(self.service.get_report(filename))
        self.assertIsNone(self.service.get_report("../../etc/passwd"))
        self.assertIsNone(self.service.get_report("other_report.json"))


class SkillEvalRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_lint_endpoint_returns_checks_and_score(self):
        response = _client(self.tmpdir).post("/api/skill-eval/lint", json={"skill_text": SKILL_TEXT})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["score"], 100)
        self.assertTrue(any(c["area"] == "security" for c in body["checks"]))

    def test_lint_endpoint_rejects_empty_payload(self):
        response = _client(self.tmpdir).post("/api/skill-eval/lint", json={})
        self.assertEqual(response.status_code, 422)

    def test_fit_endpoint_returns_verdict(self):
        response = _client(self.tmpdir).post(
            "/api/skill-eval/fit",
            json={"skill_text": SKILL_TEXT, "task_description": TASK, "judge_model": "demo-model"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "fit")
        self.assertEqual(response.json()["overall"], 0.8)

    def test_fit_endpoint_404_for_unknown_model(self):
        response = _client(self.tmpdir).post(
            "/api/skill-eval/fit",
            json={"skill_text": SKILL_TEXT, "task_description": TASK, "judge_model": "missing-model"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("missing-model", response.json()["detail"])

    def test_fit_endpoint_502_when_judge_unusable(self):
        response = _client(self.tmpdir, content="not json").post(
            "/api/skill-eval/fit",
            json={"skill_text": SKILL_TEXT, "task_description": TASK, "judge_model": "demo-model"},
        )
        self.assertEqual(response.status_code, 502)

    def test_full_endpoint_combines_and_lists_report(self):
        client = _client(self.tmpdir)
        response = client.post(
            "/api/skill-eval/full",
            json={"skill_text": SKILL_TEXT, "task_description": TASK, "judge_model": "demo-model"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["combined_basis"], "lint+fit")

        listing = client.get("/api/skill-eval/reports")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()["reports"]), 1)

        filename = listing.json()["reports"][0]["filename"]
        detail = client.get(f"/api/skill-eval/reports/{filename}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["kind"], "skill_eval")

    def test_full_endpoint_save_false_writes_nothing(self):
        client = _client(self.tmpdir)
        response = client.post(
            "/api/skill-eval/full",
            json={
                "skill_text": SKILL_TEXT,
                "task_description": TASK,
                "judge_model": "demo-model",
                "save": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/api/skill-eval/reports").json()["reports"], [])

    def test_report_detail_404_for_missing_file(self):
        response = _client(self.tmpdir).get("/api/skill-eval/reports/skill_eval_nope.json")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
