import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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
            raise RuntimeError("load_dataset should not run in regression golden smoke tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    return importlib.import_module("pipeline_runner")


class _FakeHallucinationEvaluator:
    def __init__(self, judge_adapter):
        self.judge_adapter = judge_adapter

    def check_hallucination(self, question, answer_text, reference):
        return {"score": 1.0}


class _FakeInstructionEvaluator:
    def __init__(self, judge_adapter):
        self.judge_adapter = judge_adapter

    def evaluate(self, instruction, response):
        return {"score": 1.0, "follows_instructions": True, "violations": []}


class _FakeModel:
    def __init__(self):
        self.model_name = "demo-model"
        self.calls = []
        self._responses = [
            '{"answer": "EVET"}',
            '{"answer": "Havale banka ici, EFT bankalar arasi transferdir."}',
            '{"answer": "Tam olarak hangi islemin iptal edilecegi sorulmalidir."}',
        ]

    def generate(self, messages, response_format=None, max_tokens=None):
        self.calls.append({"messages": messages, "response_format": response_format, "max_tokens": max_tokens})
        return {
            "content": self._responses[len(self.calls) - 1],
            "latency": 0.2 * len(self.calls),
            "usage": {"total_tokens": 10 * len(self.calls)},
        }


class _FakeJudge:
    def __init__(self):
        self.calls = []
        self._labels = [
            (1.0, "TAM_DOGRU", "Matches expected answer"),
            (0.5, "KISMEN_DOGRU", "Core meaning is correct"),
            (0.0, "YANLIS", "Needs clarification wording"),
        ]

    def evaluate(self, metric_name, question, answer_text, expected_output):
        self.calls.append(
            {
                "metric_name": metric_name,
                "question": question,
                "answer_text": answer_text,
                "expected_output": expected_output,
            }
        )
        score, label, reasoning = self._labels[len(self.calls) - 1]
        return {"score": score, "label": label, "reasoning": reasoning}


class _FakePipelineContext:
    def __init__(self):
        self.judge_adapter = object()
        self._task_registry = {}
        self.test_config = {"concurrent_items": 1}  # deterministic result order for golden asserts
        self._run = None

    def _inject_schema_instruction(self, system_prompt, schema):
        return system_prompt

    def _initialize_geval_evaluator(self):
        return None

    def _initialize_quality_evaluator(self):
        return None

    def _evaluate_geval_criterion(self, *args, **kwargs):
        return None

    def _evaluate_quality_scores(self, *args, **kwargs):
        return {}

    def _extend_avg_scores_with_nested_metrics(self, *args, **kwargs):
        return None

    def _parse_structured_output(self, content, schema):
        parsed = json.loads(content)
        return {
            "is_valid": True,
            "parsed": parsed,
            "parse_error": None,
            "schema_error": None,
        }


class RegressionGoldenSmokeTests(unittest.TestCase):
    def test_run_qa_test_preserves_golden_cases_and_builds_expected_summary(self):
        module = _load_pipeline_runner_module()
        dataset_path = Path(__file__).resolve().parent.parent / "eval_datasets/regression/golden.json"
        golden_cases = json.loads(dataset_path.read_text(encoding="utf-8"))[:3]
        fake_context = _FakePipelineContext()
        fake_model = _FakeModel()
        fake_judge = _FakeJudge()

        def fake_build_qa_metric_results(**kwargs):
            return [
                {"name": "json_correctness", "value": 1.0},
                {"name": "prompt_alignment", "value": 1.0},
            ]

        with patch.object(module, "HallucinationEvaluator", _FakeHallucinationEvaluator), patch.object(
            module, "InstructionFollowingEvaluator", _FakeInstructionEvaluator
        ), patch.object(module, "nlp_metrics_available", lambda: False), patch.object(
            module, "get_schema_for_test", lambda test_name: {"type": "object"}
        ), patch.object(module, "build_response_format", lambda schema: {"type": "json_object"}), patch.object(
            module, "_build_json_correctness_metric", lambda structured: {"name": "json_correctness", "value": 1.0, "raw_payload": {"is_valid": structured["is_valid"]}}
        ), patch.object(
            module, "_build_prompt_alignment_metric", lambda payload: {"name": "prompt_alignment", "value": payload["score"], "raw_payload": payload}
        ), patch.object(
            module, "_build_qa_metric_results", fake_build_qa_metric_results
        ), patch.object(
            module.CategoryMetrics,
            "calculate_per_category",
            staticmethod(lambda results: {"format": {"count": 2}, "clarity": {"count": 1}}),
        ):
            result = module.EvaluationPipeline.run_qa_test(
                fake_context,
                fake_model,
                golden_cases,
                fake_judge,
                "regression_golden",
            )

        self.assertEqual(result["test_name"], "regression_golden")
        self.assertEqual(result["summary"]["total_tests"], 3)
        self.assertEqual(result["summary"]["label_distribution"]["TAM_DOGRU"], 1)
        self.assertEqual(result["summary"]["label_distribution"]["KISMEN_DOGRU"], 1)
        self.assertEqual(result["summary"]["label_distribution"]["YANLIS"], 1)
        self.assertEqual(result["summary"]["label_distribution"]["tam_dogru_rate"], 0.333)
        self.assertEqual(result["summary"]["label_distribution"]["kismen_dogru_rate"], 0.333)
        self.assertEqual(result["summary"]["label_distribution"]["yanlis_rate"], 0.333)
        self.assertEqual(result["summary"]["overall_score"], 0.5)
        self.assertEqual(result["summary"]["schema_fail_rate"], 0)
        self.assertEqual(result["summary"]["avg_scores"]["json_correctness"], 1.0)
        self.assertEqual(result["summary"]["avg_scores"]["prompt_alignment"], 1.0)
        self.assertAlmostEqual(result["summary"]["avg_latency"], 0.4)

        self.assertEqual(result["results"][0]["id"], golden_cases[0]["id"])
        self.assertEqual(result["results"][0]["expected_answer"], golden_cases[0]["expected_answer"])
        self.assertEqual(result["results"][1]["scores"]["judge_label"], "KISMEN_DOGRU")
        self.assertEqual(result["results"][2]["scores"]["judge_score"], 0.0)
        self.assertTrue(all(item["structured_output"]["is_valid"] for item in result["results"]))
        self.assertEqual(len(fake_judge.calls), 3)
        self.assertEqual(fake_judge.calls[0]["expected_output"], golden_cases[0]["expected_answer"])
        self.assertEqual(fake_judge.calls[2]["metric_name"], "accuracy")


if __name__ == "__main__":
    unittest.main()