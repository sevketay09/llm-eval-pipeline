import json
import unittest
from pathlib import Path

import yaml


class RegressionGoldenContractsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent
        self.dataset_path = self.root / "eval_datasets/regression/golden.json"
        self.registry_path = self.root / "config/task_registry.yaml"
        self.tests_config_path = self.root / "config/tests.yaml"

    def test_golden_dataset_has_stable_deterministic_coverage(self):
        golden_cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(golden_cases), 30)

        ids = [case.get("id") for case in golden_cases]
        self.assertEqual(len(ids), len(set(ids)))

        required_categories = {
            "format",
            "safety",
            "privacy",
            "reasoning",
            "tool_selection",
            "instruction_priority",
        }
        categories = {str(case.get("category") or "").strip() for case in golden_cases}
        self.assertTrue(required_categories.issubset(categories))

        banned_phrases = (
            "it depends",
            "not specified",
            "consult documentation",
            "depends on the context",
            "duruma gore",
        )
        allowed_difficulties = {"easy", "medium", "hard"}

        for case in golden_cases:
            self.assertTrue(str(case.get("question") or "").strip())
            self.assertTrue(str(case.get("expected_answer") or "").strip())
            self.assertIn(case.get("difficulty"), allowed_difficulties)

            expected_answer = str(case.get("expected_answer") or "").strip().lower()
            self.assertFalse(any(phrase in expected_answer for phrase in banned_phrases))

    def test_golden_dataset_is_wired_into_registry_and_suites(self):
        registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        tests_config = yaml.safe_load(self.tests_config_path.read_text(encoding="utf-8")) or {}

        golden_task = (((registry or {}).get("tasks") or {}).get("regression_golden") or {})
        self.assertEqual(golden_task.get("dataset"), "eval_datasets/regression/golden.json")
        self.assertEqual(golden_task.get("runner"), "run_qa_test")
        self.assertEqual(golden_task.get("category"), "regression")

        suites = (tests_config.get("test_suites") or {})
        advanced_tests = ((suites.get("advanced") or {}).get("tests") or [])
        regression_tests = ((suites.get("regression") or {}).get("tests") or [])

        self.assertIn("regression_golden", advanced_tests)
        self.assertIn("regression_golden", regression_tests)


if __name__ == "__main__":
    unittest.main()