import importlib
import sys
import types
import unittest

from utils.case_models import ConversationTurn, MultiTurnConversationCase, SingleTurnCase


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
            raise RuntimeError("load_dataset should not run in dataset normalization tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    return importlib.import_module("pipeline_runner")


def run_multi_turn_test():
    return None


class DatasetNormalizationContractTests(unittest.TestCase):
    def test_custom_generated_falls_back_to_single_turn_and_skips_invalid_items(self):
        module = _load_pipeline_runner_module()

        prebuilt_case = SingleTurnCase(
            case_id="typed-case",
            input_text="Typed question",
            expected_output="Typed answer",
            metadata={"source": "typed"},
            raw_payload={"question": "Typed question", "expected_answer": "Typed answer"},
        )
        dataset = [
            {"id": "case-1", "question": "Merhaba", "expected_answer": "Selam", "source": "synthetic"},
            prebuilt_case,
            42,
            {"id": "broken-case"},
        ]

        normalized = module.EvaluationPipeline._normalize_dataset_for_test(
            object(),
            dataset,
            test_name="custom_generated",
            test_func=None,
        )

        self.assertEqual(len(normalized), 2)
        self.assertTrue(all(isinstance(item, SingleTurnCase) for item in normalized))
        self.assertEqual(normalized[0].case_id, "case-1")
        self.assertEqual(normalized[0].input_text, "Merhaba")
        self.assertEqual(normalized[0].expected_output, "Selam")
        self.assertEqual(normalized[0].metadata["source"], "synthetic")
        self.assertIs(normalized[1], prebuilt_case)

    def test_multi_turn_runner_normalizes_valid_payloads_and_preserves_typed_cases(self):
        module = _load_pipeline_runner_module()

        prebuilt_conversation = MultiTurnConversationCase(
            case_id="typed-conv",
            turns=[ConversationTurn(role="user", content="Ready", expected_actions=[], check=None, metadata={}, raw_payload={"role": "user", "content": "Ready"})],
            category="ops",
            difficulty="medium",
            metadata={},
            raw_payload={"turns": [{"role": "user", "content": "Ready"}]},
        )
        dataset = [
            {
                "id": "conv-1",
                "turns": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "category": "support",
                "difficulty": "easy",
                "source": "generated",
            },
            prebuilt_conversation,
            {"id": "conv-bad", "turns": []},
        ]

        normalized = module.EvaluationPipeline._normalize_dataset_for_test(
            object(),
            dataset,
            test_name="multi_turn_custom",
            test_func=run_multi_turn_test,
        )

        self.assertEqual(len(normalized), 2)
        self.assertTrue(all(isinstance(item, MultiTurnConversationCase) for item in normalized))
        self.assertEqual(normalized[0].case_id, "conv-1")
        self.assertEqual(len(normalized[0].turns), 2)
        self.assertEqual(normalized[0].turns[0].role, "user")
        self.assertEqual(normalized[0].turns[1].content, "Hi there")
        self.assertEqual(normalized[0].metadata["source"], "generated")
        self.assertIs(normalized[1], prebuilt_conversation)


if __name__ == "__main__":
    unittest.main()