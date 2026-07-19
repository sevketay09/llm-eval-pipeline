"""Contract tests: position-bias mitigation via A/B swap in ComparativeEvaluator."""
import json
import unittest

from evaluators.comparative_eval import ComparativeEvaluator


class ScriptedJudge:
    """Returns queued winner verdicts in call order."""

    def __init__(self, winners):
        self.winners = list(winners)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        winner = self.winners.pop(0)
        return {"content": json.dumps({
            "winner": winner,
            "reasoning": f"verdict {winner}",
            "score_difference": 4,
        })}


class CompareWithSwapTests(unittest.TestCase):
    def test_consistent_verdict_stands(self):
        # Pass 1: A wins. Pass 2 (swapped): judge says B — which is A in the original frame.
        judge = ScriptedJudge(["A", "B"])
        result = ComparativeEvaluator(judge).compare_with_swap("soru", "yanit-a", "yanit-b")
        self.assertEqual(result["winner"], "A")
        self.assertTrue(result["position_consistent"])
        self.assertEqual(len(judge.calls), 2)

    def test_position_biased_judge_forced_to_tie(self):
        # A biased judge always prefers the first-listed answer.
        judge = ScriptedJudge(["A", "A"])
        result = ComparativeEvaluator(judge).compare_with_swap("soru", "yanit-a", "yanit-b")
        self.assertEqual(result["winner"], "Tie")
        self.assertFalse(result["position_consistent"])
        self.assertEqual(result["score_difference"], 0.0)
        self.assertEqual(result["first_pass_winner"], "A")
        self.assertEqual(result["second_pass_winner"], "B")

    def test_double_tie_is_consistent(self):
        judge = ScriptedJudge(["Tie", "Tie"])
        result = ComparativeEvaluator(judge).compare_with_swap("soru", "a", "b")
        self.assertEqual(result["winner"], "Tie")
        self.assertTrue(result["position_consistent"])

    def test_swapped_pass_receives_swapped_responses(self):
        judge = ScriptedJudge(["A", "B"])
        ComparativeEvaluator(judge).compare_with_swap("soru", "CEVAP-A", "CEVAP-B")
        first_prompt = judge.calls[0][1]["content"]
        second_prompt = judge.calls[1][1]["content"]
        # First pass: A listed under "Model A"; second pass: order reversed
        self.assertLess(first_prompt.index("CEVAP-A"), first_prompt.index("CEVAP-B"))
        self.assertLess(second_prompt.index("CEVAP-B"), second_prompt.index("CEVAP-A"))

    def test_consistent_score_difference_averaged(self):
        judge = ScriptedJudge(["B", "A"])  # B wins both frames
        result = ComparativeEvaluator(judge).compare_with_swap("soru", "a", "b")
        self.assertEqual(result["winner"], "B")
        self.assertAlmostEqual(result["score_difference"], 0.4)  # (4/10 + 4/10) / 2


if __name__ == "__main__":
    unittest.main()
