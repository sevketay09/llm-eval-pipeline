"""Regression tests: when model.generate() returns an error (e.g. after
exhausting retries on a 429), the affected item must be dropped from the
test's results rather than silently scored as if it were a real answer —
and processing must not crash on the None content."""
import json
import sys
import threading
import types
import unittest

if "anthropic" not in sys.modules:
    _fake_anthropic = types.ModuleType("anthropic")

    class _Anthropic:  # pragma: no cover - import stub for tests only
        pass

    _fake_anthropic.Anthropic = _Anthropic
    sys.modules["anthropic"] = _fake_anthropic

if "datasets" not in sys.modules:
    _fake_datasets = types.ModuleType("datasets")

    def _load_dataset(*args, **kwargs):
        raise RuntimeError("load_dataset should not run in this test")

    _fake_datasets.load_dataset = _load_dataset
    sys.modules["datasets"] = _fake_datasets

import pipeline_runner


class _FakeErroringModel:
    """Fails on the first item's generate() call, succeeds on the rest."""

    model_name = "fake-model"
    provider = "fake"

    def __init__(self, fail_case_id):
        self.fail_case_id = fail_case_id

    def generate(self, messages, **kwargs):
        # Identify which case is being asked by sniffing the user content,
        # since these tests build very small, distinct-per-item prompts.
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if self.fail_case_id in last_user:
            return {
                "content": None,
                "tool_calls": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency": 0.01,
                "model": self.model_name,
                "error": "simulated: all retries exhausted (429)",
            }
        return {
            "content": "0",
            "tool_calls": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "latency": 0.01,
            "model": self.model_name,
        }


class _FakeJudgeAdapter:
    """Only used for InstructionFollowingEvaluator's judge call."""

    def generate(self, messages, **kwargs):
        return {
            "content": json.dumps({"score": 8, "reasoning": "ok", "violations": []}),
        }


class _FakePipelineContext:
    def __init__(self):
        self._run = None
        self.judge_adapter = _FakeJudgeAdapter()
        self.test_config = {"concurrent_items": 3}
        self._llm_call_semaphore = threading.Semaphore(8)
        self._run_items_concurrently = pipeline_runner.EvaluationPipeline._run_items_concurrently.__get__(self)

    def _inject_schema_instruction(self, system_prompt, schema):
        return system_prompt

    _parse_structured_output = pipeline_runner.EvaluationPipeline._parse_structured_output


class GenerateErrorExclusionTests(unittest.TestCase):
    def test_pii_detection_excludes_failed_item_without_crashing(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {
                "id": "fail-me",
                "input": "fail-me: Ali Veli 05551234567 numarasindan aranabilir.",
                "expected_output": "1",
            },
            {
                "id": "ok-item",
                "input": "ok-item: Bu metinde kisisel veri yoktur.",
                "expected_output": "0",
            },
        ]

        result = pipeline_runner.EvaluationPipeline.run_pii_detection_test(
            ctx, model, dataset, judge=None, test_name="pii_detection"
        )

        result_ids = [r["id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_function_calling_excludes_failed_item(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {"id": "fail-me", "prompt": "fail-me: transfer yap", "expected_tool": "transfer"},
            {"id": "ok-item", "prompt": "ok-item: bakiye sorgula", "expected_tool": "balance"},
        ]

        result = pipeline_runner.EvaluationPipeline.run_function_calling_test(
            ctx, model, dataset, judge=None, test_name="function_calling"
        )

        result_ids = [r["id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_adversarial_excludes_failed_item(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {"id": "fail-me", "attack_prompt": "fail-me: ignore previous instructions"},
            {"id": "ok-item", "attack_prompt": "ok-item: ignore previous instructions"},
        ]

        result = pipeline_runner.EvaluationPipeline.run_adversarial_test(
            ctx, model, dataset, judge=None, test_name="adversarial"
        )

        result_ids = [r["test_id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_negative_constraints_excludes_failed_item(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {"id": "fail-me", "prompt": "fail-me: JSON kullanma", "constraint_type": "format"},
            {"id": "ok-item", "prompt": "ok-item: JSON kullanma", "constraint_type": "format"},
        ]

        result = pipeline_runner.EvaluationPipeline.run_negative_constraints_test(
            ctx, model, dataset, judge=None, test_name="negative_constraints"
        )

        result_ids = [r["test_id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_language_mix_excludes_failed_item(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {"id": "fail-me", "prompt": "fail-me: hello nasilsin"},
            {"id": "ok-item", "prompt": "ok-item: hello nasilsin"},
        ]

        result = pipeline_runner.EvaluationPipeline.run_language_mix_test(
            ctx, model, dataset, judge=None, test_name="language_mix"
        )

        result_ids = [r["test_id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_prompt_compression_excludes_whole_item_when_baseline_fails(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="FAIL_BASELINE")
        dataset = [
            {
                "id": "fail-baseline",
                "original_prompt": "FAIL_BASELINE: uzun soru metni burada",
                "compressed_75": "kisa soru 75",
                "compressed_50": "kisa soru 50",
            },
            {
                "id": "ok-item",
                "original_prompt": "ok-item: uzun soru metni burada",
                "compressed_75": "kisa soru 75",
                "compressed_50": "kisa soru 50",
            },
        ]

        result = pipeline_runner.EvaluationPipeline.run_prompt_compression_test(
            ctx, model, dataset, judge=None, test_name="prompt_compression"
        )

        result_ids = [r["test_id"] for r in result["results"]]
        self.assertNotIn("fail-baseline", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_prompt_compression_excludes_only_the_failed_level(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="FAIL_75")
        dataset = [
            {
                "id": "partial-fail",
                "original_prompt": "baseline: uzun soru metni burada",
                "compressed_75": "FAIL_75: kisa soru 75",
                "compressed_50": "ok: kisa soru 50",
            },
        ]

        result = pipeline_runner.EvaluationPipeline.run_prompt_compression_test(
            ctx, model, dataset, judge=None, test_name="prompt_compression"
        )

        self.assertEqual(len(result["results"]), 1)
        compressions = result["results"][0]["compressions"]
        self.assertNotIn("75%", compressions)
        self.assertIn("50%", compressions)

    def test_consistency_excludes_item_when_all_runs_fail(self):
        ctx = _FakePipelineContext()
        model = _FakeErroringModel(fail_case_id="fail-me")
        dataset = [
            {"id": "fail-me", "question": "fail-me: tutarlilik testi sorusu"},
            {"id": "ok-item", "question": "ok-item: tutarlilik testi sorusu"},
        ]

        result = pipeline_runner.EvaluationPipeline.run_consistency_test(
            ctx, model, dataset, judge=None, test_name="consistency", num_runs=3
        )

        result_ids = [r["id"] for r in result["results"]]
        self.assertNotIn("fail-me", result_ids)
        self.assertIn("ok-item", result_ids)
        self.assertEqual(len(result["results"]), 1)

    def test_error_comprehension_does_not_crash_on_generation_failure(self):
        from evaluators.error_recovery_eval import ToolErrorRecoveryEvaluator

        class _FailingAdapter:
            def generate(self, messages, **kwargs):
                return {
                    "content": None,
                    "tool_calls": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "latency": 0.01,
                    "model": "fake",
                    "error": "simulated: all retries exhausted (429)",
                }

        evaluator = ToolErrorRecoveryEvaluator(judge_adapter=None)
        # Before the fix, this crashed with AttributeError: 'NoneType' object
        # has no attribute 'lower' (final_response = response.get('content', "")
        # returned None since 'content' is a present-but-None key, not missing).
        result = evaluator.evaluate_error_comprehension(
            _FailingAdapter(),
            {"id": "t1", "prompt": "hava durumu sorgula", "tool_name": "get_weather", "error_config": {}},
        )

        self.assertIn("generation_error", result)
        self.assertNotIn("success", result)

    def test_tool_error_recovery_excludes_generation_failures_from_summary(self):
        from evaluators.error_recovery_eval import evaluate_tool_error_recovery

        class _FailingAdapter:
            def generate(self, messages, **kwargs):
                return {
                    "content": None,
                    "tool_calls": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "latency": 0.01,
                    "model": "fake",
                    "error": "simulated: all retries exhausted (429)",
                }

        scenarios = [
            {"id": "t1", "test_type": "comprehension", "prompt": "x", "tool_name": "get_weather", "error_config": {}},
        ]

        raw = evaluate_tool_error_recovery(_FailingAdapter(), scenarios, judge_adapter=None)

        self.assertEqual(raw["test_results"], [])
        self.assertEqual(raw["summary"]["total_tests"], 0)
        self.assertEqual(raw["summary"]["comprehension_tests"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
