import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_evaluate_api_module():
    fake_pipeline_runner = types.ModuleType("pipeline_runner")

    def make_test_func(name):
        def _test_func(*args):
            return {
                "name": name,
                "arg_count": len(args),
                "args": args,
            }

        _test_func.__name__ = name
        return _test_func

    class FakeModelAdapter:
        def __init__(self, model_key):
            self.model_key = model_key
            self.reset_called = False

        def reset_stats(self):
            self.reset_called = True

    class FakeEvaluationPipeline:
        instances = []

        def __init__(self, config_path, judge_model_key=None, runtime_overrides=None):
            self.config_path = config_path
            self.judge_model_key = judge_model_key
            self.runtime_overrides = runtime_overrides
            self.calls = []
            self.saved_paths = []
            self.loaded_datasets = []
            self.initialized_models = []
            self.judge_init_count = 0
            self.__class__.instances.append(self)

        def run_full_evaluation(self, **kwargs):
            self.calls.append(("full", kwargs))
            return {"mode": "full", "kwargs": kwargs}

        def run_full_evaluation_parallel(self, **kwargs):
            self.calls.append(("parallel", kwargs))
            return {"mode": "parallel", "kwargs": kwargs}

        def run_custom_dataset_evaluation(self, **kwargs):
            self.calls.append(("custom", kwargs))
            return {"mode": "custom", "kwargs": kwargs}

        def save_results(self, output_path):
            self.saved_paths.append(output_path)

        def _build_test_mapping(self):
            return {
                "embedding_similarity": ("datasets/embedding.json", make_test_func("embedding_eval")),
                "turkish_grammar": ("datasets/grammar.json", make_test_func("run_qa_test")),
            }

        def load_dataset(self, dataset_path, max_samples, test_name=None, test_func=None):
            dataset = {
                "dataset_path": dataset_path,
                "max_samples": max_samples,
                "test_name": test_name,
                "test_func_name": getattr(test_func, "__name__", None),
            }
            self.loaded_datasets.append(dataset)
            return dataset

        def initialize_model(self, model):
            adapter = FakeModelAdapter(model)
            self.initialized_models.append(adapter)
            return adapter

        def initialize_judge(self):
            self.judge_init_count += 1
            return {"kind": "judge", "count": self.judge_init_count}

    fake_pipeline_runner.EvaluationPipeline = FakeEvaluationPipeline
    sys.modules["pipeline_runner"] = fake_pipeline_runner

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

    module_path = Path(__file__).resolve().parent / "evaluate_api.py"
    spec = importlib.util.spec_from_file_location("isolated_evaluate_api", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, FakeEvaluationPipeline


class EvaluateApiSmokeTests(unittest.TestCase):
    def test_parallel_route_preserves_selected_tests_and_saves_output(self):
        module, FakeEvaluationPipeline = _load_evaluate_api_module()
        FakeEvaluationPipeline.instances.clear()

        result = module.evaluate(
            models=["demo-a", "demo-b"],
            suite="smoke",
            tests=["case_a"],
            judge_model="judge-x",
            output_path="reports/smoke.json",
            parallel=True,
            max_workers=3,
            temperature=0.2,
            top_p=0.8,
            max_tokens=1024,
        )

        pipeline = FakeEvaluationPipeline.instances[0]
        self.assertEqual(pipeline.judge_model_key, "judge-x")
        self.assertEqual(pipeline.runtime_overrides, {"temperature": 0.2, "top_p": 0.8, "max_tokens": 1024})
        self.assertEqual(pipeline.calls[0][0], "parallel")
        self.assertEqual(pipeline.calls[0][1]["selected_tests"], ["case_a"])
        self.assertEqual(pipeline.calls[0][1]["max_workers"], 3)
        self.assertEqual(pipeline.saved_paths, ["reports/smoke.json"])
        self.assertEqual(result["mode"], "parallel")

    def test_custom_dataset_route_wins_over_suite_and_parallel_flags(self):
        module, FakeEvaluationPipeline = _load_evaluate_api_module()
        FakeEvaluationPipeline.instances.clear()

        result = module.evaluate(
            models=["demo-a"],
            suite="smoke",
            tests=["case_a"],
            output_path="reports/custom.json",
            parallel=True,
            max_workers=2,
            custom_dataset_path="datasets/generated.json",
            custom_dataset_name="Generated QA",
            custom_dataset_kind="conversation",
        )

        pipeline = FakeEvaluationPipeline.instances[0]
        self.assertEqual(pipeline.calls[0][0], "custom")
        self.assertEqual(pipeline.calls[0][1]["dataset_path"], "datasets/generated.json")
        self.assertEqual(pipeline.calls[0][1]["dataset_name"], "Generated QA")
        self.assertEqual(pipeline.calls[0][1]["dataset_kind"], "conversation")
        self.assertTrue(pipeline.calls[0][1]["parallel"])
        self.assertEqual(pipeline.saved_paths, ["reports/custom.json"])
        self.assertEqual(result["mode"], "custom")

    def test_evaluate_single_skips_judge_for_embedding_tests(self):
        module, FakeEvaluationPipeline = _load_evaluate_api_module()
        FakeEvaluationPipeline.instances.clear()

        result = module.evaluate_single(
            model="demo-embed",
            test="embedding_similarity",
            max_samples=4,
            judge_model="judge-x",
            temperature=0.15,
        )

        pipeline = FakeEvaluationPipeline.instances[0]
        self.assertEqual(pipeline.runtime_overrides, {"temperature": 0.15})
        self.assertEqual(pipeline.loaded_datasets[0]["dataset_path"], "datasets/embedding.json")
        self.assertEqual(pipeline.loaded_datasets[0]["max_samples"], 4)
        self.assertEqual(pipeline.judge_init_count, 0)
        self.assertTrue(pipeline.initialized_models[0].reset_called)
        self.assertEqual(result["model"], "demo-embed")
        self.assertEqual(result["test"], "embedding_similarity")
        self.assertEqual(result["result"]["arg_count"], 3)
        self.assertEqual(result["result"]["name"], "embedding_eval")

    def test_evaluate_single_initializes_judge_for_standard_tests(self):
        module, FakeEvaluationPipeline = _load_evaluate_api_module()
        FakeEvaluationPipeline.instances.clear()

        result = module.evaluate_single(
            model="demo-llm",
            test="turkish_grammar",
            max_samples=2,
        )

        pipeline = FakeEvaluationPipeline.instances[0]
        self.assertEqual(pipeline.loaded_datasets[0]["dataset_path"], "datasets/grammar.json")
        self.assertEqual(pipeline.judge_init_count, 1)
        self.assertEqual(result["result"]["arg_count"], 4)
        self.assertEqual(result["result"]["name"], "run_qa_test")

    def test_evaluate_single_rejects_unknown_test_names(self):
        module, FakeEvaluationPipeline = _load_evaluate_api_module()
        FakeEvaluationPipeline.instances.clear()

        with self.assertRaises(ValueError) as raised:
            module.evaluate_single(model="demo-llm", test="does_not_exist")

        self.assertIn("Unknown test 'does_not_exist'", str(raised.exception))


if __name__ == "__main__":
    unittest.main()