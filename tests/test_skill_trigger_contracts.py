import json
import unittest

from analysis.skill_trigger import (
    SkillTriggerChecker,
    extract_skill_meta,
    routing_metrics,
    summarize,
)

SKILL = """---
name: csv-report
description: Generates weekly CSV sales reports with totals per region.
---
# Usage
Load the CSV, group by region, write totals.
"""


class KeywordAdapter:
    """Triggers when the prompt mentions csv/report/sales — deterministic stand-in."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on or set()

    def generate(self, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        prompt = messages[1]["content"].lower()
        if any(marker in prompt for marker in self.fail_on):
            return {"content": "no json here", "latency": 0.1, "usage": {}}
        trigger = any(word in prompt for word in ("csv", "report", "sales", "satış", "rapor"))
        return {"content": json.dumps({"trigger": trigger}), "latency": 0.1, "usage": {}}


PROMPTS = [
    {"text": "Generate the weekly sales CSV report", "expected": True},
    {"text": "Build the regional sales report from data.csv", "expected": True},
    {"text": "What is the weather in Ankara?", "expected": False},
    {"text": "Translate this sentence to German", "expected": False},
    {"text": "Summarize this spreadsheet somehow", "expected": "ambiguous"},
]


class ExtractMetaTests(unittest.TestCase):
    def test_extracts_name_and_description(self):
        meta = extract_skill_meta(SKILL)
        self.assertEqual(meta["name"], "csv-report")
        self.assertIn("CSV sales reports", meta["description"])

    def test_missing_frontmatter_gives_empty_meta(self):
        self.assertEqual(extract_skill_meta("# body only"), {"name": "", "description": ""})


class RoutingMetricsTests(unittest.TestCase):
    def test_perfect_routing(self):
        results = [
            {"expected": True, "predicted": True},
            {"expected": True, "predicted": True},
            {"expected": False, "predicted": False},
            {"expected": False, "predicted": False},
        ]
        metrics = routing_metrics(results)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_false_positive_lowers_precision_not_recall(self):
        results = [
            {"expected": True, "predicted": True},
            {"expected": False, "predicted": True},
            {"expected": False, "predicted": False},
            {"expected": True, "predicted": True},
        ]
        metrics = routing_metrics(results)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertLess(metrics["precision"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.5)

    def test_ambiguous_and_unparsed_excluded(self):
        results = [
            {"expected": "ambiguous", "predicted": True},
            {"expected": True, "predicted": None},
            {"expected": True, "predicted": True},
        ]
        self.assertEqual(routing_metrics(results)["scored"], 1)


class SummarizeTests(unittest.TestCase):
    def _results(self, tp=2, fp=0, fn=0, tn=2):
        out = []
        out += [{"expected": True, "predicted": True}] * tp
        out += [{"expected": False, "predicted": True}] * fp
        out += [{"expected": True, "predicted": False}] * fn
        out += [{"expected": False, "predicted": False}] * tn
        return out

    def test_reliable_verdict(self):
        self.assertEqual(summarize(self._results())["verdict"], "reliable")

    def test_over_triggering_verdict(self):
        summary = summarize(self._results(tp=2, fp=2, tn=1))
        self.assertEqual(summary["verdict"], "over_triggering")

    def test_under_triggering_verdict(self):
        summary = summarize(self._results(tp=1, fn=2, tn=2))
        self.assertEqual(summary["verdict"], "under_triggering")

    def test_insufficient_data_below_min_prompts(self):
        self.assertEqual(summarize(self._results(tp=1, tn=1))["verdict"], "insufficient_data")

    def test_ambiguous_trigger_rate_reported_separately(self):
        results = self._results() + [
            {"expected": "ambiguous", "predicted": True},
            {"expected": "ambiguous", "predicted": False},
        ]
        summary = summarize(results)
        self.assertEqual(summary["ambiguous_count"], 2)
        self.assertEqual(summary["ambiguous_trigger_rate"], 0.5)
        self.assertEqual(summary["scored"], 4)


class SkillTriggerCheckerTests(unittest.TestCase):
    def test_end_to_end_reliable_routing(self):
        checker = SkillTriggerChecker(KeywordAdapter())
        report = checker.run(SKILL, PROMPTS)
        self.assertEqual(report["skill"]["name"], "csv-report")
        self.assertEqual(report["summary"]["verdict"], "reliable")
        self.assertEqual(report["summary"]["precision"], 1.0)
        self.assertEqual(report["summary"]["recall"], 1.0)
        self.assertEqual(len(report["results"]), 5)

    def test_probe_prompt_includes_description_not_body(self):
        adapter = KeywordAdapter()
        SkillTriggerChecker(adapter).run(SKILL, PROMPTS[:1])
        system = adapter.calls[0][0]["content"]
        self.assertIn("csv-report", system)
        self.assertIn("CSV sales reports", system)
        self.assertNotIn("group by region", system)  # body must stay hidden

    def test_unparseable_reply_skips_prompt_without_guessing(self):
        adapter = KeywordAdapter(fail_on={"weather"})
        report = SkillTriggerChecker(adapter).run(SKILL, PROMPTS)
        weather = next(r for r in report["results"] if "weather" in r["text"].lower())
        self.assertIsNone(weather["predicted"])
        self.assertEqual(report["summary"]["skipped"], 1)

    def test_repeats_majority_vote_and_trial_count(self):
        checker = SkillTriggerChecker(KeywordAdapter(), repeats=3)
        report = checker.run(SKILL, PROMPTS[:1])
        result = report["results"][0]
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["trigger_rate"], 1.0)
        self.assertTrue(result["predicted"])

    def test_malformed_prompts_are_skipped(self):
        checker = SkillTriggerChecker(KeywordAdapter())
        report = checker.run(SKILL, [{"text": "", "expected": True}, {"text": "x", "expected": "maybe"}])
        self.assertEqual(report["results"], [])
        self.assertEqual(report["summary"]["verdict"], "insufficient_data")

    def test_adapter_exception_does_not_kill_run(self):
        class ExplodingAdapter(KeywordAdapter):
            def generate(self, messages, response_format=None, max_tokens=None):
                if "weather" in messages[1]["content"].lower():
                    raise RuntimeError("boom")
                return super().generate(messages, response_format, max_tokens)

        report = SkillTriggerChecker(ExplodingAdapter()).run(SKILL, PROMPTS)
        self.assertEqual(len(report["results"]), 4)  # exploded prompt dropped


if __name__ == "__main__":
    unittest.main()
