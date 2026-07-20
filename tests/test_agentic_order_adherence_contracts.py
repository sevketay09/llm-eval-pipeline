"""Contract tests for pipeline_runner._evaluate_agentic_tool_call_order (plan/sequence adherence).

Closes a gap versus tool_selection (set-based, order-blind): mcp_tool_use /
agentic_workflows multi-tool cases previously had no signal for whether the
expected tools were called in the expected order, unlike function_calling_chain.
"""
import importlib
import sys
import types
import unittest


def _load_pipeline_runner_module():
    fake_anthropic = sys.modules.get("anthropic")
    if fake_anthropic is None:
        fake_anthropic = types.ModuleType("anthropic")

        class Anthropic:  # pragma: no cover - import stub for tests only
            pass

        fake_anthropic.Anthropic = Anthropic
        sys.modules["anthropic"] = fake_anthropic

    fake_datasets = sys.modules.get("datasets")
    if fake_datasets is None:
        fake_datasets = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            raise RuntimeError("load_dataset should not run in order-adherence tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    return importlib.import_module("pipeline_runner")


pipeline_runner = _load_pipeline_runner_module()
_evaluate_agentic_tool_call_order = pipeline_runner._evaluate_agentic_tool_call_order
_build_agentic_order_adherence_metric = pipeline_runner._build_agentic_order_adherence_metric


def _calls(*names):
    return [{"name": n, "arguments": {}} for n in names]


class EvaluateAgenticToolCallOrderTests(unittest.TestCase):
    def test_none_for_single_expected_tool(self):
        self.assertIsNone(_evaluate_agentic_tool_call_order(["get_weather"], _calls("get_weather")))

    def test_none_for_empty_expected_tools(self):
        self.assertIsNone(_evaluate_agentic_tool_call_order([], _calls("get_weather")))

    def test_perfect_order_scores_one(self):
        summary = _evaluate_agentic_tool_call_order(
            ["get_risk_profile", "get_market_data", "calculate_portfolio_allocation"],
            _calls("get_risk_profile", "get_market_data", "calculate_portfolio_allocation"),
        )
        self.assertEqual(summary["score"], 1.0)
        self.assertTrue(summary["exact_match"])
        self.assertEqual(summary["matched_in_order"], 3)

    def test_reversed_order_scores_low(self):
        summary = _evaluate_agentic_tool_call_order(
            ["get_risk_profile", "get_market_data"],
            _calls("get_market_data", "get_risk_profile"),
        )
        self.assertLess(summary["score"], 1.0)
        self.assertFalse(summary["exact_match"])

    def test_extra_calls_do_not_break_subsequence_match(self):
        summary = _evaluate_agentic_tool_call_order(
            ["get_risk_profile", "get_market_data"],
            _calls("get_risk_profile", "unrelated_tool", "get_market_data"),
        )
        self.assertEqual(summary["score"], 1.0)
        self.assertTrue(summary["exact_match"])

    def test_no_tool_calls_scores_zero(self):
        summary = _evaluate_agentic_tool_call_order(["a", "b"], [])
        self.assertEqual(summary["score"], 0.0)
        self.assertFalse(summary["exact_match"])
        self.assertEqual(summary["called_sequence"], [])

    def test_partial_match_before_diverging(self):
        summary = _evaluate_agentic_tool_call_order(
            ["a", "b", "c"],
            _calls("a", "b", "x"),
        )
        self.assertEqual(summary["matched_in_order"], 2)
        self.assertEqual(summary["score"], round(2 / 3, 4))


class BuildOrderAdherenceMetricTests(unittest.TestCase):
    def test_none_when_summary_missing_score(self):
        self.assertIsNone(_build_agentic_order_adherence_metric({"reason": "no score"}))

    def test_none_when_summary_not_dict(self):
        self.assertIsNone(_build_agentic_order_adherence_metric(None))

    def test_builds_metric_with_group_tool_usage_pack(self):
        metric = _build_agentic_order_adherence_metric({"score": 1.0, "reason": "ok", "exact_match": True})
        self.assertIsNotNone(metric)
        self.assertEqual(metric["name"], "order_adherence")
        self.assertEqual(metric["group"], "tool_usage_pack")
        self.assertTrue(metric["success"])

    def test_below_threshold_marks_failure(self):
        metric = _build_agentic_order_adherence_metric({"score": 0.3, "reason": "diverged"})
        self.assertFalse(metric["success"])


if __name__ == "__main__":
    unittest.main()
