"""Contract tests: judge parse failures must yield None scores, never fake 0.0.

Covers evaluators/judge_utils.py + parse paths of groundedness/quality/agent judges.
"""
import unittest

from evaluators.judge_utils import request_judge_json, extract_score, strip_code_fences, JSON_RESPONSE_FORMAT
from evaluators.groundedness_judge import GroundednessJudgeEvaluator
from evaluators.quality_judge import QualityJudgeEvaluator
from evaluators.agent_judge import AgentJudgeEvaluator


class ScriptedAdapter:
    """Returns queued contents in order; records calls."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        content = self.contents.pop(0) if self.contents else self.contents_fallback()
        return {"content": content}

    def contents_fallback(self):
        return "{}"


class KeywordAdapter:
    """Picks response by keyword found in the user message (thread-safe for evaluate_all)."""

    def __init__(self, mapping, default='{"score": 4, "reasoning": "ok"}'):
        self.mapping = mapping
        self.default = default

    def generate(self, messages, **kwargs):
        user = " ".join(m["content"] for m in messages)
        for keyword, content in self.mapping.items():
            if keyword in user:
                return {"content": content}
        return {"content": self.default}


class RequestJudgeJsonTests(unittest.TestCase):
    def test_valid_first_attempt_single_call(self):
        adapter = ScriptedAdapter(['{"score": 4, "reasoning": "ok"}'])
        parsed = request_judge_json(adapter, [{"role": "user", "content": "x"}], "t")
        self.assertEqual(parsed["score"], 4)
        self.assertEqual(len(adapter.calls), 1)

    def test_response_format_requested(self):
        adapter = ScriptedAdapter(['{"score": 4}'])
        request_judge_json(adapter, [{"role": "user", "content": "x"}], "t")
        self.assertEqual(adapter.calls[0]["kwargs"].get("response_format"), JSON_RESPONSE_FORMAT)

    def test_retry_recovers_on_second_attempt(self):
        adapter = ScriptedAdapter(["not json", '{"score": 3}'])
        parsed = request_judge_json(adapter, [{"role": "user", "content": "x"}], "t")
        self.assertEqual(parsed["score"], 3)
        self.assertEqual(len(adapter.calls), 2)

    def test_all_attempts_fail_returns_none(self):
        adapter = ScriptedAdapter(["garbage", "still garbage"])
        self.assertIsNone(request_judge_json(adapter, [{"role": "user", "content": "x"}], "t"))
        self.assertEqual(len(adapter.calls), 2)

    def test_json_array_rejected(self):
        adapter = ScriptedAdapter(["[1, 2]", "[3]"])
        self.assertIsNone(request_judge_json(adapter, [{"role": "user", "content": "x"}], "t"))

    def test_code_fences_stripped(self):
        self.assertEqual(strip_code_fences('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(strip_code_fences('```\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(strip_code_fences('{"a": 1}'), '{"a": 1}')


class ExtractScoreTests(unittest.TestCase):
    def test_numeric_and_string_numeric(self):
        self.assertEqual(extract_score({"score": 4}, "t"), 4.0)
        self.assertEqual(extract_score({"score": "3.5"}, "t"), 3.5)

    def test_missing_or_invalid_returns_none(self):
        self.assertIsNone(extract_score({}, "t"))
        self.assertIsNone(extract_score({"score": "yüksek"}, "t"))
        self.assertIsNone(extract_score(None, "t"))


class GroundednessJudgeParseTests(unittest.TestCase):
    def test_parse_failure_yields_none_score(self):
        adapter = ScriptedAdapter(["bozuk", "yine bozuk"])
        result = GroundednessJudgeEvaluator(adapter).evaluate("yanit", "baglam")
        self.assertIsNone(result["score"])
        self.assertIsNone(result["normalized_score"])
        self.assertIsNone(result["is_faithful"])
        self.assertEqual(result["result"], "error")

    def test_valid_verdict(self):
        adapter = ScriptedAdapter(['{"score": 4, "reasoning": "ok", "result": "pass"}'])
        result = GroundednessJudgeEvaluator(adapter).evaluate("yanit", "baglam", "soru")
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["normalized_score"], 0.8)
        self.assertTrue(result["is_faithful"])


class QualityJudgeParseTests(unittest.TestCase):
    def test_failed_metric_omitted_from_evaluate_all(self):
        # fluency prompt mentions "dil kalitesini"; make only that one fail (2 attempts both bad)
        adapter = KeywordAdapter({"dil kalitesini": "bozuk json"})
        scores = QualityJudgeEvaluator(adapter).evaluate_all(query="q", response="r", context="c")
        self.assertNotIn("fluency", scores)
        for name in ("coherence", "relevance", "groundedness"):
            self.assertEqual(scores[name], 4.0)

    def test_single_metric_parse_failure_returns_none(self):
        adapter = ScriptedAdapter(["bozuk", "bozuk"])
        result = QualityJudgeEvaluator(adapter).evaluate_coherence("q", "r")
        self.assertIsNone(result["score"])
        self.assertIsNone(result["normalized"])


class AgentJudgeParseTests(unittest.TestCase):
    def test_error_excluded_from_aggregate(self):
        # intent prompt uniquely contains "temel amacı"; fail only that metric
        adapter = KeywordAdapter(
            {"temel amacı": "bozuk json"},
            default='{"score": 5, "reasoning": "ok", "result": "pass"}',
        )
        result = AgentJudgeEvaluator(adapter).evaluate_all(query="q", response="r")
        self.assertIsNone(result["intent_resolution"]["score"])
        self.assertEqual(result["aggregate_score"], 1.0)  # 3 valid metrics, all 5/5

    def test_all_errors_aggregate_none(self):
        adapter = KeywordAdapter({}, default="bozuk json")
        result = AgentJudgeEvaluator(adapter).evaluate_all(query="q", response="r")
        self.assertIsNone(result["aggregate_score"])

    def test_genuine_zero_counts_in_aggregate(self):
        adapter = KeywordAdapter({}, default='{"score": 0, "reasoning": "kötü", "result": "fail"}')
        result = AgentJudgeEvaluator(adapter).evaluate_all(query="q", response="r")
        self.assertEqual(result["aggregate_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
