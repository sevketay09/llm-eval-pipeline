import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

from utils.report_renderer import render_html_report, render_markdown_report, render_terminal_summary


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
            raise RuntimeError("load_dataset should not run in regression suite smoke tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    return importlib.import_module("pipeline_runner")


class _FakeModelRunResult:
    @staticmethod
    def empty(model_key, model_name, provider, runtime_parameters):
        payload = {
            "model_key": model_key,
            "model_name": model_name,
            "provider": provider,
            "runtime_parameters": runtime_parameters,
            "tests": {},
            "overall_metrics": {},
        }

        class _PayloadWrapper:
            def to_payload(self_inner):
                return payload

        return _PayloadWrapper()


class _FakeModel:
    def __init__(self, model_key):
        self.model_key = model_key
        self.model_name = f"model::{model_key}"
        self.provider = "fake"
        self.reset_calls = 0

    def reset_stats(self):
        self.reset_calls += 1


class _FakePipelineContext:
    def __init__(self):
        self.config_path = "config/models.yaml"
        self.runtime_overrides = {"temperature": 0.1}
        self.results = {
            "timestamp": "2026-05-31T18:00:00Z",
            "run_metadata": {},
            "models": {},
        }
        self.test_config = {
            "test_suites": {
                "regression": {
                    "tests": ["regression_golden", "regression_recent"],
                    "max_samples": "all",
                }
            }
        }
        self.execution_log = []
        self.loaded_datasets = []
        self.judge = object()
        self.judge_init_count = 0
        self.models = []
        self.saved_paths = []
        self.final_saved_exports = []
        self._run = None
        self._tqdm_position_by_thread = threading.local()

    def _attach_ai_commentaries(self, model_keys):
        return None

    def _build_test_mapping(self):
        return {
            "regression_golden": ("eval_datasets/regression/golden.json", self._make_runner("regression_golden")),
            "regression_recent": ("eval_datasets/regression/recent_issues.json", self._make_runner("regression_recent")),
            "turkish_grammar": ("eval_datasets/benchmark/turkish_grammar.json", self._make_runner("turkish_grammar")),
        }

    def _make_runner(self, test_name):
        def _runner(model, dataset, judge, called_test_name):
            self.execution_log.append(
                {
                    "test_name": called_test_name,
                    "dataset": dataset,
                    "judge": judge,
                    "model": model.model_key,
                }
            )
            return {
                "test_name": called_test_name,
                "summary": {"overall_score": 0.8 if test_name.endswith("golden") else 0.6},
                "results": [{"id": f"{called_test_name}-1"}],
            }

        return _runner

    def run_qa_test(self, model, dataset, judge, called_test_name):
        return self._make_runner("custom_generated")(model, dataset, judge, called_test_name)

    def run_multi_turn_test(self, model, dataset, judge, called_test_name):
        return self._make_runner("custom_generated_conversation")(model, dataset, judge, called_test_name)

    def initialize_judge(self):
        self.judge_init_count += 1
        return self.judge

    def initialize_model(self, model_key):
        model = _FakeModel(model_key)
        self.models.append(model)
        return model

    def load_dataset(self, dataset_path, max_samples, test_name=None, test_func=None):
        dataset = {
            "dataset_path": dataset_path,
            "max_samples": max_samples,
            "test_name": test_name,
        }
        self.loaded_datasets.append(dataset)
        return dataset

    def _update_model_overall_metrics(self, model, model_results):
        model_results["overall_metrics"]["weighted_score"] = len(model_results["tests"])

    def _generate_summary(self):
        return {
            "model_comparison": {
                model_key: {"overall_score": len(model_payload.get("tests", {}))}
                for model_key, model_payload in self.results["models"].items()
            }
        }

    def _generate_trends(self, model_keys):
        return {model_key: {"history_points": 0} for model_key in model_keys}

    def _generate_comparisons(self, model_keys):
        return {"models": list(model_keys)}

    def save_results(self, output_path, quiet=False, render_reports=True):
        self.saved_paths.append({"output_path": output_path, "quiet": quiet, "render_reports": render_reports})

    def _build_custom_model_results(self, model_key, model):
        return _FakeModelRunResult.empty(
            model_key=model_key,
            model_name=model.model_name,
            provider=model.provider,
            runtime_parameters=dict(self.runtime_overrides),
        ).to_payload()


def _attach_custom_dataset_helpers(module, fake_context):
    fake_context._run_custom_dataset_sequential = types.MethodType(
        module.EvaluationPipeline._run_custom_dataset_sequential, fake_context
    )
    fake_context._run_custom_dataset_parallel = types.MethodType(
        module.EvaluationPipeline._run_custom_dataset_parallel, fake_context
    )


class RegressionSuiteSmokeTests(unittest.TestCase):
    def test_run_full_evaluation_executes_regression_suite_in_order(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()

        with patch.object(module, "capture_config_snapshot", lambda **kwargs: {"run_id": "run-reg-1"}), patch.object(
            module, "ModelRunResult", _FakeModelRunResult
        ), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ), patch.object(
            module, "hash_results", lambda payload: "hash-reg-1"
        ):
            result = module.EvaluationPipeline.run_full_evaluation(
                fake_context,
                model_keys=["demo-model"],
                test_suite="regression",
            )

        self.assertEqual(result["run_metadata"]["test_suite"], "regression")
        self.assertEqual(result["run_metadata"]["selected_tests"], [])
        self.assertEqual(result["run_metadata"]["run_id"], "run-reg-1")
        self.assertEqual(result["run_metadata"]["result_hash"], "hash-reg-1")
        self.assertEqual(fake_context.judge_init_count, 1)
        self.assertEqual([entry["test_name"] for entry in fake_context.execution_log], ["regression_golden", "regression_recent"])
        self.assertEqual(
            [item["dataset_path"] for item in fake_context.loaded_datasets],
            ["eval_datasets/regression/golden.json", "eval_datasets/regression/recent_issues.json"],
        )
        self.assertEqual(fake_context.models[0].reset_calls, 1)
        self.assertIn("regression_golden", result["models"]["demo-model"]["tests"])
        self.assertIn("regression_recent", result["models"]["demo-model"]["tests"])
        self.assertEqual(result["summary"]["model_comparison"]["demo-model"]["overall_score"], 2)
        self.assertEqual(result["trends"]["demo-model"]["history_points"], 0)

    def test_run_full_evaluation_honors_selected_tests_within_suite_order(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()

        with patch.object(module, "capture_config_snapshot", lambda **kwargs: {"run_id": "run-reg-2"}), patch.object(
            module, "ModelRunResult", _FakeModelRunResult
        ), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ), patch.object(
            module, "hash_results", lambda payload: "hash-reg-2"
        ):
            result = module.EvaluationPipeline.run_full_evaluation(
                fake_context,
                model_keys=["demo-model"],
                test_suite="regression",
                selected_tests=["regression_recent"],
            )

        self.assertEqual(result["run_metadata"]["selected_tests"], ["regression_recent"])
        self.assertEqual([entry["test_name"] for entry in fake_context.execution_log], ["regression_recent"])
        self.assertEqual(list(result["models"]["demo-model"]["tests"].keys()), ["regression_recent"])
        self.assertEqual(result["summary"]["model_comparison"]["demo-model"]["overall_score"], 1)

    def test_run_full_evaluation_parallel_fans_out_regression_suite_per_model(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()

        with patch.object(module, "ModelRunResult", _FakeModelRunResult), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ):
            result = module.EvaluationPipeline.run_full_evaluation_parallel(
                fake_context,
                model_keys=["demo-a", "demo-b"],
                test_suite="regression",
                selected_tests=["regression_recent"],
                output_path=None,
                max_workers=2,
            )

        self.assertTrue(result["run_metadata"]["parallel_models"])
        self.assertEqual(result["run_metadata"]["test_suite"], "regression")
        self.assertEqual(result["run_metadata"]["selected_tests"], ["regression_recent"])
        self.assertEqual(fake_context.judge_init_count, 1)
        self.assertEqual(len(fake_context.loaded_datasets), 1)
        self.assertEqual(fake_context.loaded_datasets[0]["test_name"], "regression_recent")
        self.assertEqual(sorted(entry["model"] for entry in fake_context.execution_log), ["demo-a", "demo-b"])
        self.assertEqual([entry["test_name"] for entry in fake_context.execution_log], ["regression_recent", "regression_recent"])
        self.assertEqual(list(result["models"]["demo-a"]["tests"].keys()), ["regression_recent"])
        self.assertEqual(list(result["models"]["demo-b"]["tests"].keys()), ["regression_recent"])
        self.assertEqual(result["summary"]["model_comparison"]["demo-a"]["overall_score"], 1)
        self.assertEqual(result["summary"]["model_comparison"]["demo-b"]["overall_score"], 1)
        self.assertEqual(result["trends"]["demo-a"]["history_points"], 0)
        self.assertEqual(result["trends"]["demo-b"]["history_points"], 0)

    def test_run_full_evaluation_payload_renders_across_all_export_formats(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()

        with patch.object(module, "capture_config_snapshot", lambda **kwargs: {"run_id": "run-reg-export"}), patch.object(
            module, "ModelRunResult", _FakeModelRunResult
        ), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ), patch.object(
            module, "hash_results", lambda payload: "hash-reg-export"
        ):
            result = module.EvaluationPipeline.run_full_evaluation(
                fake_context,
                model_keys=["demo-model"],
                test_suite="regression",
                selected_tests=["regression_golden"],
            )

        terminal_output = render_terminal_summary(result)
        markdown_output = render_markdown_report(result)
        html_output = render_html_report(result)

        self.assertIn("Run Metadata:", terminal_output)
        self.assertIn("Suite: regression", terminal_output)
        self.assertIn("Selected Tests: regression_golden", terminal_output)
        self.assertIn("demo-model:", terminal_output)

        self.assertIn("## Run Metadata", markdown_output)
        self.assertIn("- Suite: regression", markdown_output)
        self.assertIn("- Selected Tests: regression_golden", markdown_output)
        self.assertIn("| demo-model | 1.000 |", markdown_output)

        self.assertIn("<h2>Run Metadata</h2>", html_output)
        self.assertIn("<td>regression</td>", html_output)
        self.assertIn("<td>regression_golden</td>", html_output)
        self.assertIn("<h2>Model Comparison</h2>", html_output)
        self.assertIn("demo-model", html_output)

    def test_run_custom_dataset_evaluation_routes_conversation_dataset_to_multi_turn_runner(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()
        _attach_custom_dataset_helpers(module, fake_context)

        def load_custom_dataset(dataset_path, max_samples, test_name=None, test_func=None):
            fake_context.loaded_datasets.append(
                {
                    "dataset_path": dataset_path,
                    "max_samples": max_samples,
                    "test_name": test_name,
                    "test_func_name": getattr(test_func, "__name__", None),
                }
            )
            return [{"id": "conv-1"}, {"id": "conv-2"}]

        fake_context.load_dataset = load_custom_dataset

        with patch.object(module, "capture_config_snapshot", lambda **kwargs: {"run_id": "run-custom-conv"}), patch.object(
            module, "ModelRunResult", _FakeModelRunResult
        ), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ), patch.object(
            module, "hash_results", lambda payload: "hash-custom-conv"
        ), patch.object(
            module, "save_reproducible_results", lambda payload, snapshot, output_path: fake_context.final_saved_exports.append(output_path)
        ):
            result = module.EvaluationPipeline.run_custom_dataset_evaluation(
                fake_context,
                model_keys=["demo-model"],
                dataset_path="tmp/generated-conversation.json",
                dataset_name="Conversation QA",
                dataset_kind="conversation",
                output_path="reports/custom-conversation.json",
                parallel=False,
            )

        self.assertEqual(result["run_metadata"]["test_suite"], "custom_generated_conversation")
        self.assertEqual(result["run_metadata"]["run_id"], "run-custom-conv")
        self.assertEqual(result["run_metadata"]["result_hash"], "hash-custom-conv")
        self.assertEqual(
            result["run_metadata"]["custom_dataset"],
            {
                "name": "Conversation QA",
                "path": "tmp/generated-conversation.json",
                "dataset_kind": "conversation",
                "item_count": 2,
            },
        )
        self.assertEqual(fake_context.judge_init_count, 1)
        self.assertEqual(fake_context.loaded_datasets[0]["test_name"], "custom_generated_conversation")
        self.assertEqual(fake_context.loaded_datasets[0]["test_func_name"], "run_multi_turn_test")
        self.assertEqual([entry["test_name"] for entry in fake_context.execution_log], ["custom_generated_conversation"])
        self.assertIn("custom_generated_conversation", result["models"]["demo-model"]["tests"])
        self.assertEqual(fake_context.saved_paths, [{"output_path": "reports/custom-conversation.json", "quiet": True, "render_reports": False}])
        self.assertEqual(fake_context.final_saved_exports, ["reports/custom-conversation.json"])

    def test_run_custom_dataset_evaluation_parallel_routes_single_turn_dataset_to_qa_runner(self):
        module = _load_pipeline_runner_module()
        fake_context = _FakePipelineContext()
        _attach_custom_dataset_helpers(module, fake_context)

        def load_custom_dataset(dataset_path, max_samples, test_name=None, test_func=None):
            fake_context.loaded_datasets.append(
                {
                    "dataset_path": dataset_path,
                    "max_samples": max_samples,
                    "test_name": test_name,
                    "test_func_name": getattr(test_func, "__name__", None),
                }
            )
            return [{"id": "qa-1"}]

        fake_context.load_dataset = load_custom_dataset

        with patch.object(module, "capture_config_snapshot", lambda **kwargs: {"run_id": "run-custom-qa"}), patch.object(
            module, "ModelRunResult", _FakeModelRunResult
        ), patch.object(
            module, "serialize_test_result_payload", lambda test_name, payload: payload
        ), patch.object(
            module, "_annotate_test_result_payload_metadata", lambda test_name, dataset_path, payload: {**payload, "dataset_path": dataset_path}
        ), patch.object(
            module, "serialize_run_payload", lambda payload: payload
        ), patch.object(
            module, "hash_results", lambda payload: "hash-custom-qa"
        ), patch.object(
            module, "save_reproducible_results", lambda payload, snapshot, output_path: fake_context.final_saved_exports.append(output_path)
        ):
            result = module.EvaluationPipeline.run_custom_dataset_evaluation(
                fake_context,
                model_keys=["demo-a", "demo-b"],
                dataset_path="tmp/generated-qa.json",
                dataset_kind="single_turn",
                output_path="reports/custom-qa.json",
                parallel=True,
                max_workers=2,
            )

        self.assertEqual(result["run_metadata"]["test_suite"], "custom_generated")
        self.assertEqual(result["run_metadata"]["custom_dataset"]["dataset_kind"], "single_turn")
        self.assertEqual(result["run_metadata"]["custom_dataset"]["item_count"], 1)
        self.assertEqual(fake_context.judge_init_count, 1)
        self.assertEqual(fake_context.loaded_datasets[0]["test_name"], "custom_generated")
        self.assertEqual(fake_context.loaded_datasets[0]["test_func_name"], "run_qa_test")
        self.assertEqual(sorted(entry["model"] for entry in fake_context.execution_log), ["demo-a", "demo-b"])
        self.assertEqual([entry["test_name"] for entry in fake_context.execution_log], ["custom_generated", "custom_generated"])
        self.assertIn("custom_generated", result["models"]["demo-a"]["tests"])
        self.assertIn("custom_generated", result["models"]["demo-b"]["tests"])
        self.assertEqual(result["summary"]["model_comparison"]["demo-a"]["overall_score"], 1)
        self.assertEqual(result["summary"]["model_comparison"]["demo-b"]["overall_score"], 1)
        self.assertEqual(
            fake_context.saved_paths,
            [
                {"output_path": "reports/custom-qa.json", "quiet": True, "render_reports": False},
                {"output_path": "reports/custom-qa.json", "quiet": True, "render_reports": False},
            ],
        )
        self.assertEqual(fake_context.final_saved_exports, ["reports/custom-qa.json"])


if __name__ == "__main__":
    unittest.main()