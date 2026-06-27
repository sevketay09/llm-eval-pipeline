import json
import tempfile
import unittest
from pathlib import Path

from api.services.report_service import ReportService
from utils.report_renderer import render_html_report, render_markdown_report, render_terminal_summary


class ReportVersioningContractTests(unittest.TestCase):
    def test_export_renderers_include_version_metadata(self):
        payload = {
            "version": "2.0",
            "timestamp": "2026-05-31T10:00:00Z",
            "run_metadata": {
                "run_id": "versioned1234",
                "test_suite": "smoke",
                "prompt_version": "judge-v3",
                "judge_prompt_version": "judge-v3",
                "schema_version": "2.0",
                "metric_version": "metric-pack-bundle-v1",
                "config_checksum": "abc",
                "tests_config_checksum": "def",
            },
            "models": {
                "demo": {
                    "tests": {
                        "smoke_case": {
                            "summary": {"overall_score": 0.9}
                        }
                    }
                }
            },
            "summary": {"model_comparison": {"demo": {"overall_score": 0.9}}},
            "trends": {},
        }

        terminal_output = render_terminal_summary(payload)
        markdown_output = render_markdown_report(payload)
        html_output = render_html_report(payload)

        self.assertIn("Prompt Version: judge-v3", terminal_output)
        self.assertIn("Schema Version: 2.0", terminal_output)
        self.assertIn("Metric Version: metric-pack-bundle-v1", terminal_output)
        self.assertIn("- Prompt Version: judge-v3", markdown_output)
        self.assertIn("- Schema Version: 2.0", markdown_output)
        self.assertIn("- Metric Version: metric-pack-bundle-v1", markdown_output)
        self.assertIn("Prompt Version", html_output)
        self.assertIn("judge-v3", html_output)
        self.assertIn("Schema Version", html_output)
        self.assertIn("Metric Version", html_output)

    def test_legacy_report_metadata_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "legacy-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "timestamp": "2026-05-31T10:00:00Z",
                        "run_metadata": {
                            "run_id": "legacy1234",
                            "test_suite": "smoke",
                        },
                        "models": {
                            "demo": {
                                "tests": {
                                    "smoke_case": {
                                        "summary": {"overall_score": 0.8},
                                    }
                                }
                            }
                        },
                        "summary": {"model_comparison": {"demo": {"overall_score": 0.8}}},
                        "trends": {},
                    }
                ),
                encoding="utf-8",
            )

            service = ReportService()
            service._dir = Path(tmpdir)

            report = service.get_report("legacy-report.json")

            self.assertIsNotNone(report)
            self.assertEqual(report.metadata["prompt_version"], "legacy")
            self.assertEqual(report.metadata["judge_prompt_version"], "legacy")
            self.assertEqual(report.metadata["schema_version"], "2.0")
            self.assertEqual(report.metadata["metric_version"], "legacy")
            self.assertEqual(report.metadata["metric_pack_versions"], {})

    def test_compare_reports_includes_version_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "versioned-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "version": "2.0",
                        "timestamp": "2026-05-31T10:00:00Z",
                        "run_metadata": {
                            "run_id": "run1234",
                            "test_suite": "smoke",
                            "prompt_version": "judge-v3",
                            "judge_prompt_version": "judge-v3",
                            "schema_version": "2.0",
                            "metric_version": "metric-pack-bundle-v1",
                            "metric_pack_versions": {"safety_metric_pack": "v2"},
                        },
                        "models": {
                            "demo": {
                                "tests": {
                                    "smoke_case": {
                                        "summary": {"overall_score": 0.9},
                                    }
                                }
                            }
                        },
                        "summary": {"model_comparison": {"demo": {"overall_score": 0.9}}},
                        "trends": {},
                    }
                ),
                encoding="utf-8",
            )

            service = ReportService()
            service._dir = Path(tmpdir)

            compare = service.compare_reports(["versioned-report.json"])

            self.assertEqual(compare["versioned-report.json"]["prompt_version"], "judge-v3")
            self.assertEqual(compare["versioned-report.json"]["schema_version"], "2.0")
            self.assertEqual(compare["versioned-report.json"]["metric_version"], "metric-pack-bundle-v1")


if __name__ == "__main__":
    unittest.main()