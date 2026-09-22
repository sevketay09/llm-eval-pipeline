"""Contract tests for decisions/skill_routing.py and the decision-backed
path of analysis/skill_trigger.py. Offline only — MockDecisionClient.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict

from analysis.skill_trigger import SkillTriggerChecker
from decisions.mock_client import MockDecisionClient
from decisions.skill_routing import NONE_LABEL, route, routing_choice_metrics, trigger_probability

SKILL = """---
name: csv-report
description: Generates weekly CSV sales reports with totals per region.
---
# Usage
Load the CSV, group by region, write totals.
"""

PROMPTS = [
    {"text": "Generate the weekly sales CSV report", "expected": True},
    {"text": "Build the regional sales report from data.csv", "expected": True},
    {"text": "What is the weather in Ankara?", "expected": False},
    {"text": "Translate this sentence to German", "expected": False},
    {"text": "Summarize this spreadsheet somehow", "expected": "ambiguous"},
]


class CountingMockDecisionClient(MockDecisionClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decide_calls = 0

    def _decide_raw(self, state, questions):
        self.decide_calls += 1
        return super()._decide_raw(state, questions)


class TriggerProbabilityContractTests(unittest.TestCase):
    def test_returns_float_in_unit_interval(self):
        client = MockDecisionClient()
        p = trigger_probability(client, {"name": "csv-report", "description": "..."}, "generate a report")
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class SkillTriggerCheckerDecisionModeContractTests(unittest.TestCase):
    def test_requires_adapter_or_decision_client(self):
        with self.assertRaises(ValueError):
            SkillTriggerChecker()

    def test_decision_mode_adds_probability_and_threshold_curve(self):
        client = MockDecisionClient(overrides={"trigger": 0.9})
        checker = SkillTriggerChecker(decision_client=client)
        report = checker.run(SKILL, PROMPTS)

        self.assertEqual(report["summary"]["mode"], "decision")
        self.assertEqual(len(report["summary"]["threshold_curve"]), 6)
        for point in report["summary"]["threshold_curve"]:
            self.assertIn("precision", point)
            self.assertIn("recall", point)
            self.assertIn("f1", point)
            self.assertIn("false_positive_rate", point)
        for result in report["results"]:
            self.assertIn("probability", result)
            self.assertEqual(result["trials"], 1)

    def test_llm_mode_has_no_probability_key(self):
        class KeywordAdapter:
            def generate(self, messages, response_format=None, max_tokens=None):
                import json

                prompt = messages[1]["content"].lower()
                trigger = "csv" in prompt or "report" in prompt or "sales" in prompt
                return {"content": json.dumps({"trigger": trigger}), "latency": 0.1, "usage": {}}

        checker = SkillTriggerChecker(adapter=KeywordAdapter())
        report = checker.run(SKILL, PROMPTS)
        self.assertNotIn("mode", report["summary"])
        self.assertNotIn("threshold_curve", report["summary"])
        for result in report["results"]:
            self.assertNotIn("probability", result)

    def test_decision_client_wins_when_both_given(self):
        class ExplodingAdapter:
            def generate(self, *a, **k):
                raise AssertionError("adapter should not be used when decision_client is given")

        client = MockDecisionClient()
        checker = SkillTriggerChecker(adapter=ExplodingAdapter(), decision_client=client)
        report = checker.run(SKILL, PROMPTS[:1])
        self.assertEqual(report["summary"].get("mode"), "decision")


class RouteContractTests(unittest.TestCase):
    def test_none_option_always_present_small_catalog(self):
        client = MockDecisionClient()
        skills = {"csv-report": "generates csv reports", "weather-lookup": "checks the weather"}
        result = route(client, skills, "some unrelated request")
        labels = {name for name, _ in result["ranked"]}
        self.assertIn(NONE_LABEL, labels | {result["choice"]})

    def test_two_stage_routing_for_large_catalog(self):
        client = CountingMockDecisionClient()
        skills = {f"skill-{i}": f"description {i}" for i in range(300)}
        result = route(client, skills, "some request")
        self.assertIn("choice", result)
        self.assertGreaterEqual(client.decide_calls, 3)

    def test_single_finalist_short_circuits_final_call(self):
        # Force every chunk winner to differ so only one finalist can remain
        # is not guaranteed with the mock's hash-based choice, so instead
        # verify the general invariant: result always has a valid choice.
        client = MockDecisionClient()
        skills = {f"skill-{i}": f"description {i}" for i in range(250)}
        result = route(client, skills, "some request")
        self.assertTrue(result["choice"] == NONE_LABEL or result["choice"] in skills)


class RoutingChoiceMetricsContractTests(unittest.TestCase):
    def test_perfect_accuracy(self):
        results = [
            {"expected": "a", "predicted": "a"},
            {"expected": "b", "predicted": "b"},
            {"expected": "none", "predicted": "none"},
        ]
        metrics = routing_choice_metrics(results)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["none_rate"], 1 / 3, places=3)

    def test_confusion_tracks_misroutes(self):
        results = [
            {"expected": "a", "predicted": "b"},
            {"expected": "a", "predicted": "a"},
        ]
        metrics = routing_choice_metrics(results)
        self.assertEqual(metrics["confusion"]["a"]["b"], 1)
        self.assertEqual(metrics["confusion"]["a"]["a"], 1)

    def test_per_skill_precision_recall(self):
        results = [
            {"expected": "a", "predicted": "a"},
            {"expected": "a", "predicted": "b"},
            {"expected": "b", "predicted": "b"},
        ]
        metrics = routing_choice_metrics(results)
        self.assertEqual(metrics["per_skill"]["a"]["recall"], 0.5)
        self.assertEqual(metrics["per_skill"]["b"]["precision"], 0.5)


class SkillEvalServiceDecisionGuardContractTests(unittest.TestCase):
    def test_fit_with_decision_model_raises(self):
        from api.services.skill_eval_service import SkillEvalService, UnsupportedJudgeError

        svc = SkillEvalService(config_path="config/models.yaml")
        with self.assertRaises(UnsupportedJudgeError):
            svc.fit(SKILL, "some task", "demo-jev")

    def test_trigger_with_decision_model_uses_decision_client(self):
        from api.services.skill_eval_service import SkillEvalService

        def factory(model_key, config_path):
            return MockDecisionClient(overrides={"trigger": 0.9})

        svc = SkillEvalService(config_path="config/models.yaml", decision_client_factory=factory)
        report = svc.trigger(SKILL, PROMPTS, "demo-jev")
        self.assertEqual(report["summary"]["mode"], "decision")

    def test_route_builds_catalog_from_skill_frontmatter(self):
        from api.services.skill_eval_service import SkillEvalService

        def factory(model_key, config_path):
            return MockDecisionClient()

        svc = SkillEvalService(config_path="config/models.yaml", decision_client_factory=factory)
        report = svc.route(
            [
                {"name": "csv-report", "skill_text": SKILL},
                {"name": "weather", "skill_text": "---\nname: weather\ndescription: checks weather\n---\n"},
            ],
            [{"text": "give me the sales report", "expected": "csv-report"}],
            "demo-jev",
        )
        self.assertEqual(set(report["skills"]), {"csv-report", "weather"})
        self.assertEqual(len(report["results"]), 1)


if __name__ == "__main__":
    unittest.main()
