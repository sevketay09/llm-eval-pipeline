import json
import unittest

from evaluators.skill_fit_judge import (
    CRITERIA,
    FIT_THRESHOLD,
    PARTIAL_FIT_THRESHOLD,
    SkillFitJudge,
)

SKILL_TEXT = """---
name: csv-report
description: Generates weekly CSV sales reports with totals per region.
---
# Usage
Load the CSV, group by region, write totals to report.csv.
"""

TASK = "Haftalık satış CSV'sinden bölge bazlı toplam raporu üret."


def _verdict_payload(score=0.9, evidence="Load the CSV, group by region"):
    return {
        "criteria": {
            name: {"score": score, "evidence": evidence, "reasoning": "ok"}
            for name in CRITERIA
        },
        "gaps": ["No error handling for malformed rows"],
        "suggestions": ["Add a delimiter option"],
    }


class StubAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, response_format=None, max_tokens=None):
        self.calls.append({"messages": messages, "response_format": response_format})
        content = self.responses.pop(0) if self.responses else "{}"
        return {"content": content, "latency": 0.1, "usage": {"total_tokens": 5}}


class SkillFitJudgeContractTests(unittest.TestCase):
    def test_full_verdict_parsed_with_all_criteria(self):
        judge = SkillFitJudge(StubAdapter([json.dumps(_verdict_payload(0.8))]))
        report = judge.evaluate(SKILL_TEXT, TASK)
        self.assertEqual(report["overall"], 0.8)
        self.assertEqual(sorted(report["criteria"]), sorted(CRITERIA))
        self.assertEqual(report["missing_criteria"], [])
        self.assertEqual(report["gaps"], ["No error handling for malformed rows"])
        self.assertEqual(report["suggestions"], ["Add a delimiter option"])

    def test_verdict_bands(self):
        for score, expected in ((0.9, "fit"), (0.6, "partial_fit"), (0.2, "unfit")):
            judge = SkillFitJudge(StubAdapter([json.dumps(_verdict_payload(score))]))
            self.assertEqual(judge.evaluate(SKILL_TEXT, TASK)["verdict"], expected)

    def test_band_edges_are_inclusive(self):
        for score, expected in ((FIT_THRESHOLD, "fit"), (PARTIAL_FIT_THRESHOLD, "partial_fit")):
            judge = SkillFitJudge(StubAdapter([json.dumps(_verdict_payload(score))]))
            self.assertEqual(judge.evaluate(SKILL_TEXT, TASK)["verdict"], expected)

    def test_scores_clamped_to_unit_range(self):
        payload = _verdict_payload()
        payload["criteria"]["scope_coverage"]["score"] = 3.7
        payload["criteria"]["completeness"]["score"] = -1.0
        judge = SkillFitJudge(StubAdapter([json.dumps(payload)]))
        report = judge.evaluate(SKILL_TEXT, TASK)
        self.assertEqual(report["criteria"]["scope_coverage"]["score"], 1.0)
        self.assertEqual(report["criteria"]["completeness"]["score"], 0.0)

    def test_non_numeric_criterion_dropped_and_listed_missing(self):
        payload = _verdict_payload(0.8)
        payload["criteria"]["efficiency_risk"]["score"] = "high"
        judge = SkillFitJudge(StubAdapter([json.dumps(payload)]))
        report = judge.evaluate(SKILL_TEXT, TASK)
        self.assertNotIn("efficiency_risk", report["criteria"])
        self.assertEqual(report["missing_criteria"], ["efficiency_risk"])
        self.assertEqual(report["overall"], 0.8)  # mean over remaining four

    def test_evidence_preserved_and_null_allowed(self):
        payload = _verdict_payload()
        payload["criteria"]["scope_coverage"]["evidence"] = None
        judge = SkillFitJudge(StubAdapter([json.dumps(payload)]))
        report = judge.evaluate(SKILL_TEXT, TASK)
        self.assertIsNone(report["criteria"]["scope_coverage"]["evidence"])
        self.assertEqual(
            report["criteria"]["instruction_clarity"]["evidence"],
            "Load the CSV, group by region",
        )

    def test_parse_failure_returns_none_not_zero(self):
        judge = SkillFitJudge(StubAdapter(["not json at all", "still not json"]))
        self.assertIsNone(judge.evaluate(SKILL_TEXT, TASK))

    def test_missing_criteria_object_returns_none(self):
        judge = SkillFitJudge(StubAdapter([json.dumps({"gaps": []})]))
        self.assertIsNone(judge.evaluate(SKILL_TEXT, TASK))

    def test_all_criteria_unusable_returns_none(self):
        payload = {"criteria": {name: {"score": "n/a"} for name in CRITERIA}}
        judge = SkillFitJudge(StubAdapter([json.dumps(payload)]))
        self.assertIsNone(judge.evaluate(SKILL_TEXT, TASK))

    def test_empty_inputs_return_none_without_calling_judge(self):
        adapter = StubAdapter([json.dumps(_verdict_payload())])
        judge = SkillFitJudge(adapter)
        self.assertIsNone(judge.evaluate("", TASK))
        self.assertIsNone(judge.evaluate(SKILL_TEXT, "   "))
        self.assertEqual(adapter.calls, [])

    def test_prompt_contains_task_and_skill_and_requests_json(self):
        adapter = StubAdapter([json.dumps(_verdict_payload())])
        SkillFitJudge(adapter).evaluate(SKILL_TEXT, TASK)
        call = adapter.calls[0]
        user_msg = call["messages"][1]["content"]
        self.assertIn(TASK, user_msg)
        self.assertIn("csv-report", user_msg)
        self.assertEqual(call["response_format"], {"type": "json_object"})
        for name in CRITERIA:
            self.assertIn(name, call["messages"][0]["content"])

    def test_retry_once_then_parse_success(self):
        adapter = StubAdapter(["garbage", json.dumps(_verdict_payload(0.7))])
        report = SkillFitJudge(adapter).evaluate(SKILL_TEXT, TASK)
        self.assertEqual(report["overall"], 0.7)
        self.assertEqual(len(adapter.calls), 2)

    def test_gaps_and_suggestions_default_to_empty_lists(self):
        payload = _verdict_payload()
        payload.pop("gaps")
        payload["suggestions"] = "not-a-list"
        report = SkillFitJudge(StubAdapter([json.dumps(payload)])).evaluate(SKILL_TEXT, TASK)
        self.assertEqual(report["gaps"], [])
        self.assertEqual(report["suggestions"], [])


if __name__ == "__main__":
    unittest.main()
