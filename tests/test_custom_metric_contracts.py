"""Contract tests for evaluators.custom_metric — offline, no real LLM."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluators.custom_metric import (
    _parse_judge_response,
    _pearson_correlation,
    _render_prompt,
    calibrate_metric,
    evaluate_with_custom_metric,
    generate_judge_prompt,
    load_metric,
    save_metric,
)


class GenerateJudgePromptContractTests(unittest.TestCase):
    """Test generate_judge_prompt function."""

    def test_template_based_no_llm(self):
        """No llm_fn: result contains literal {question}, {answer}, {expected_answer}."""
        result = generate_judge_prompt("Test description")
        self.assertIn("{question}", result)
        self.assertIn("{answer}", result)
        self.assertIn("{expected_answer}", result)

    def test_empty_description_still_returns_prompt(self):
        """Empty description returns non-empty prompt."""
        result = generate_judge_prompt("")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_with_llm_fn_good_response(self):
        """With llm_fn returning good prompt, uses it."""
        def mock_llm(messages):
            return "SORU: {question}\nVERİ: {answer}\nBEKLENEN: {expected_answer}\nScore: 1.0"

        result = generate_judge_prompt("Test", llm_fn=mock_llm)
        self.assertIn("{question}", result)
        self.assertIn("{answer}", result)
        self.assertIn("{expected_answer}", result)

    def test_prompt_contains_score_instruction(self):
        """Prompt contains score or puan instruction."""
        result = generate_judge_prompt("test")
        self.assertTrue("score" in result.lower() or "puan" in result.lower())


class RenderPromptContractTests(unittest.TestCase):
    """Test _render_prompt function."""

    def test_fills_all_placeholders(self):
        """All placeholders filled from case dict."""
        prompt = "Q: {question} A: {answer} E: {expected_answer}"
        case = {
            "question": "What is 2+2?",
            "answer": "4",
            "expected_answer": "4"
        }
        result = _render_prompt(prompt, case)
        self.assertEqual(result, "Q: What is 2+2? A: 4 E: 4")

    def test_missing_answer_defaults_to_empty(self):
        """Missing answer key doesn't raise, defaults to empty string."""
        prompt = "Q: {question} A: {answer}"
        case = {"question": "test"}
        result = _render_prompt(prompt, case)
        self.assertIn("Q: test A: ", result)

    def test_extra_case_fields_ignored(self):
        """Extra keys in case don't cause errors."""
        prompt = "Q: {question}"
        case = {"question": "test", "extra": "field", "other": 123}
        result = _render_prompt(prompt, case)
        self.assertEqual(result, "Q: test")


class GenerateThenRenderRoundTripContractTests(unittest.TestCase):
    """generate_judge_prompt()'s real output must survive _render_prompt() —
    prior bug: the literal {"score": ...} JSON example in the generated
    template collided with str.format_map()'s field-parsing and crashed
    every real evaluation with 'Invalid format specifier'."""

    def test_real_generated_prompt_renders_without_raising(self):
        prompt = generate_judge_prompt("Rate helpfulness from 0 to 1")
        case = {"question": "q", "answer": "a", "expected_answer": "e"}
        rendered = _render_prompt(prompt, case)
        self.assertIn("q", rendered)
        self.assertIn("a", rendered)
        # The JSON example must survive as single braces, not vanish or double.
        self.assertIn('{"score":', rendered)

    def test_description_with_braces_is_preserved_not_swallowed(self):
        """Curly braces typed by the user into the description must show up
        verbatim in the rendered prompt, not be silently eaten by the later
        format_map() render step."""
        prompt = generate_judge_prompt("Match the desired {tone} exactly")
        case = {"question": "q", "answer": "a", "expected_answer": "e"}
        rendered = _render_prompt(prompt, case)
        self.assertIn("{tone}", rendered)


class ParseJudgeResponseContractTests(unittest.TestCase):
    """Test _parse_judge_response function."""

    def test_plain_json(self):
        """Plain JSON response parsed correctly."""
        raw = '{"score": 0.8, "reasoning": "good"}'
        result = _parse_judge_response(raw)
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["reasoning"], "good")

    def test_markdown_fenced(self):
        """Markdown-fenced JSON parsed correctly."""
        raw = '```json\n{"score": 0.5, "reasoning": "ok"}\n```'
        result = _parse_judge_response(raw)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["reasoning"], "ok")

    def test_unparseable_returns_none_score(self):
        """Unparseable response returns None score."""
        raw = "not json at all"
        result = _parse_judge_response(raw)
        self.assertIsNone(result["score"])
        self.assertEqual(result["reasoning"], raw)


class EvaluateWithCustomMetricContractTests(unittest.TestCase):
    """Test evaluate_with_custom_metric function."""

    def test_valid_response(self):
        """Valid LLM response is parsed and returned."""
        def mock_llm(messages):
            return '{"score": 0.75, "reasoning": "İyi"}'

        case = {"question": "test", "answer": "good", "expected_answer": "good"}
        prompt = "Q: {question} A: {answer}"
        result = evaluate_with_custom_metric(case, prompt, llm_fn=mock_llm)

        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["reasoning"], "İyi")

    def test_score_clamped(self):
        """Score > 1.0 clamped to 1.0."""
        def mock_llm(messages):
            return '{"score": 1.5, "reasoning": "x"}'

        case = {"question": "test", "answer": "answer"}
        prompt = "test"
        result = evaluate_with_custom_metric(case, prompt, llm_fn=mock_llm)

        self.assertEqual(result["score"], 1.0)

    def test_question_in_rendered_prompt(self):
        """Case question appears in rendered prompt sent to LLM."""
        captured_messages = []
        def mock_llm(messages):
            captured_messages.extend(messages)
            return '{"score": 0.5, "reasoning": "ok"}'

        case = {"question": "What is AI?", "answer": "A system"}
        prompt = "Q: {question} A: {answer}"
        evaluate_with_custom_metric(case, prompt, llm_fn=mock_llm)

        content = captured_messages[0]["content"]
        self.assertIn("What is AI?", content)

    def test_missing_score_key_no_raise(self):
        """Missing score key doesn't raise, returns None score."""
        def mock_llm(messages):
            return '{"reasoning": "ok"}'

        case = {}
        prompt = "test"
        result = evaluate_with_custom_metric(case, prompt, llm_fn=mock_llm)

        self.assertIsNone(result["score"])


class CalibrateMetricContractTests(unittest.TestCase):
    """Test calibrate_metric function."""

    def test_perfect_alignment(self):
        """Perfect alignment: MAE ≈ 0, level='good'."""
        def mock_llm(messages):
            return '{"score": 1.0, "reasoning": "perfect"}'

        examples = [
            {"question": "q1", "expected_score": 1.0},
            {"question": "q2", "expected_score": 1.0},
            {"question": "q3", "expected_score": 1.0},
        ]

        result = calibrate_metric("test", examples, llm_fn=mock_llm)

        self.assertAlmostEqual(result["mean_absolute_error"], 0.0, places=1)
        self.assertEqual(result["alignment_level"], "good")

    def test_high_error(self):
        """High error: MAE=1.0, level='poor'."""
        def mock_llm(messages):
            return '{"score": 0.0, "reasoning": "bad"}'

        examples = [
            {"question": "q1", "expected_score": 1.0},
            {"question": "q2", "expected_score": 1.0},
            {"question": "q3", "expected_score": 1.0},
        ]

        result = calibrate_metric("test", examples, llm_fn=mock_llm)

        self.assertAlmostEqual(result["mean_absolute_error"], 1.0, places=1)
        self.assertEqual(result["alignment_level"], "poor")

    def test_per_example_count(self):
        """per_example count matches number of examples."""
        def mock_llm(messages):
            return '{"score": 0.5, "reasoning": "ok"}'

        examples = [
            {"expected_score": 0.5},
            {"expected_score": 0.6},
            {"expected_score": 0.7},
        ]

        result = calibrate_metric("test", examples, llm_fn=mock_llm)

        self.assertEqual(len(result["per_example"]), len(examples))

    def test_too_few_for_correlation(self):
        """< 3 successful predictions: correlation is None."""
        def mock_llm(messages):
            return '{"score": 0.5, "reasoning": "ok"}'

        examples = [
            {"expected_score": 0.5},
            {"expected_score": 0.6},
        ]

        result = calibrate_metric("test", examples, llm_fn=mock_llm)

        self.assertIsNone(result["correlation"])


if __name__ == "__main__":
    unittest.main()
