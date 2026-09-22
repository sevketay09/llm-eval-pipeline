"""Contract tests for decisions/ core: types, mock client, stats, cascade,
config resolution, and the Jev SDK adapter mapping (via a fake SDK module —
no real typesafe-sdk network calls).
"""
from __future__ import annotations

import sys
import types
import unittest
import unittest.mock
from pathlib import Path

from decisions.cascade import CascadePolicy, should_escalate
from decisions.client import BaseDecisionClient, DecisionStats
from decisions.config import (
    build_decision_client,
    get_model_config,
    is_decision_model,
    is_decision_provider,
    load_models_config,
)
from decisions.mock_client import MockDecisionClient
from decisions.types import ChoiceQ, DecisionError, DecisionResponse, NoulQ, ScoreQ


# ============================================================================= #
# Question type validation
# ============================================================================= #

class QuestionTypeContractTests(unittest.TestCase):
    def test_choice_requires_at_least_one_option(self):
        with self.assertRaises(ValueError):
            ChoiceQ("x", {})

    def test_choice_rejects_over_255_options(self):
        criteria = {f"opt{i}": "d" for i in range(256)}
        with self.assertRaises(ValueError):
            ChoiceQ("x", criteria)

    def test_score_requires_at_least_one_level(self):
        with self.assertRaises(ValueError):
            ScoreQ("x", [])


# ============================================================================= #
# MockDecisionClient
# ============================================================================= #

class MockDecisionClientContractTests(unittest.TestCase):
    def test_noul_deterministic_for_same_state_and_name(self):
        c1 = MockDecisionClient()
        c2 = MockDecisionClient()
        q = {"flag": NoulQ("is this true?")}
        r1 = c1.decide("same state", q)
        r2 = c2.decide("same state", q)
        self.assertEqual(r1.answers["flag"].value, r2.answers["flag"].value)

    def test_noul_override(self):
        c = MockDecisionClient(overrides={"flag": 0.95})
        r = c.decide("anything", {"flag": NoulQ("q")})
        self.assertEqual(r.answers["flag"].value, 0.95)
        self.assertEqual(r.answers["flag"].confidence, 0.95)

    def test_choice_override_picks_full_probability(self):
        c = MockDecisionClient(overrides={"intent": "loans"})
        r = c.decide("x", {"intent": ChoiceQ("q", {"loans": "d1", "cards": "d2"})})
        self.assertEqual(r.answers["intent"].value, "loans")
        self.assertAlmostEqual(r.answers["intent"].probabilities["loans"], 0.8)

    def test_score_override(self):
        c = MockDecisionClient(overrides={"quality": 0.2})
        r = c.decide("x", {"quality": ScoreQ("q", [("low", "d"), ("high", "d")])})
        self.assertEqual(r.answers["quality"].value, 0.2)

    def test_fail_times_then_recovers_within_default_retries(self):
        c = MockDecisionClient(fail_times=2)
        r = c.decide("x", {"flag": NoulQ("q")})
        self.assertIsInstance(r, DecisionResponse)
        self.assertEqual(c.stats.snapshot()["requests"], 1)

    def test_fail_times_exceeds_retries_raises_decision_error(self):
        c = MockDecisionClient(fail_times=10)
        with self.assertRaises(DecisionError):
            c.decide("x", {"flag": NoulQ("q")})
        self.assertEqual(c.stats.snapshot()["errors"], 1)


# ============================================================================= #
# DecisionStats / BaseDecisionClient
# ============================================================================= #

class DecisionStatsContractTests(unittest.TestCase):
    def test_snapshot_computes_cost_from_tokens(self):
        stats = DecisionStats(cost_per_mtok_input=1.0)
        stats.record(DecisionResponse(answers={}, latency_ms=5.0, est_input_tokens=1_000_000, model="m"))
        snap = stats.snapshot()
        self.assertEqual(snap["requests"], 1)
        self.assertAlmostEqual(snap["est_cost_usd"], 1.0)

    def test_snapshot_empty_is_zeroed(self):
        snap = DecisionStats().snapshot()
        self.assertEqual(snap["requests"], 0)
        self.assertEqual(snap["latency_p50_ms"], 0.0)


class TruncatingClient(BaseDecisionClient):
    def __init__(self, **kwargs):
        super().__init__("k", "m", max_state_chars=10, **kwargs)
        self.seen_state = None

    def _decide_raw(self, state, questions):
        self.seen_state = state
        return DecisionResponse(answers={}, latency_ms=1.0, est_input_tokens=1, model="m")


class BaseDecisionClientContractTests(unittest.TestCase):
    def test_truncates_long_state(self):
        c = TruncatingClient()
        c.decide("x" * 100, {"q": NoulQ("q")})
        self.assertTrue(c.seen_state.endswith("[...truncated]"))
        self.assertLess(len(c.seen_state), 100)

    def test_handles_own_retries_makes_single_attempt(self):
        class CountingFail(BaseDecisionClient):
            handles_own_retries = True

            def __init__(self):
                super().__init__("k", "m")
                self.calls = 0

            def _decide_raw(self, state, questions):
                self.calls += 1
                raise RuntimeError("boom")

        c = CountingFail()
        with self.assertRaises(DecisionError):
            c.decide("x", {"q": NoulQ("q")})
        self.assertEqual(c.calls, 1)


# ============================================================================= #
# CascadePolicy
# ============================================================================= #

class CascadePolicyContractTests(unittest.TestCase):
    def test_from_config_defaults(self):
        policy = CascadePolicy.from_config(None)
        self.assertEqual(policy.accept_confidence, 0.85)
        self.assertEqual(policy.hitl_below, 0.60)

    def test_from_config_overrides(self):
        policy = CascadePolicy.from_config({"cascade": {"accept_confidence": 0.9, "hitl_below": 0.5}})
        self.assertEqual(policy.accept_confidence, 0.9)
        self.assertEqual(policy.hitl_below, 0.5)

    def test_should_escalate_on_none_confidence(self):
        policy = CascadePolicy()
        self.assertTrue(should_escalate(policy, [0.9, None]))

    def test_should_escalate_on_low_confidence(self):
        policy = CascadePolicy(accept_confidence=0.8)
        self.assertTrue(should_escalate(policy, [0.9, 0.5]))

    def test_no_escalation_when_all_confident(self):
        policy = CascadePolicy(accept_confidence=0.8)
        self.assertFalse(should_escalate(policy, [0.9, 0.85]))

    def test_no_confidences_escalates(self):
        policy = CascadePolicy()
        self.assertTrue(should_escalate(policy, []))


# ============================================================================= #
# decisions.config
# ============================================================================= #

class DecisionConfigContractTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.config_path = str(Path(self._tmpdir.name) / "models.yaml")
        Path(self.config_path).write_text(
            "models:\n"
            "  demo-jev:\n"
            "    provider: typesafe-mock\n"
            "    model_name: jev-mock\n"
            "    api_key: none\n"
            "  demo-model:\n"
            "    provider: mock\n"
            "    model_name: demo-model\n"
            "    api_key: none\n"
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_is_decision_provider(self):
        self.assertTrue(is_decision_provider("typesafe"))
        self.assertTrue(is_decision_provider("typesafe-mock"))
        self.assertFalse(is_decision_provider("openai"))
        self.assertFalse(is_decision_provider(None))

    def test_load_models_config_reads_yaml(self):
        config = load_models_config(self.config_path)
        self.assertIn("demo-jev", config["models"])

    def test_get_model_config_missing_raises(self):
        with self.assertRaises(ValueError):
            get_model_config("nonexistent", self.config_path)

    def test_is_decision_model_true_and_false(self):
        self.assertTrue(is_decision_model("demo-jev", self.config_path))
        self.assertFalse(is_decision_model("demo-model", self.config_path))
        self.assertFalse(is_decision_model("nonexistent", self.config_path))

    def test_build_decision_client_mock(self):
        client = build_decision_client("demo-jev", self.config_path)
        self.assertIsInstance(client, MockDecisionClient)

    def test_build_decision_client_rejects_llm_provider(self):
        with self.assertRaises(ValueError):
            build_decision_client("demo-model", self.config_path)


# ============================================================================= #
# JevDecisionClient — SDK absent, and mapping logic via a fake SDK module
# ============================================================================= #

class JevDecisionClientContractTests(unittest.TestCase):
    def setUp(self):
        self._saved_module = sys.modules.pop("typesafe_sdk", None)

    def tearDown(self):
        if self._saved_module is not None:
            sys.modules["typesafe_sdk"] = self._saved_module
        else:
            sys.modules.pop("typesafe_sdk", None)

    def test_missing_sdk_raises_decision_error(self):
        """Forces the ImportError path regardless of whether typesafe-sdk
        happens to be installed in the current environment (dev machines
        that verified the SDK contract will have it; CI won't)."""
        import builtins

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "typesafe_sdk":
                raise ImportError("No module named 'typesafe_sdk'")
            return real_import(name, *args, **kwargs)

        from decisions.jev_client import JevDecisionClient

        client = JevDecisionClient({"api_key": "fake", "model_name": "jev-latest"}, "jev")
        with unittest.mock.patch("builtins.__import__", side_effect=blocking_import):
            with self.assertRaises(DecisionError) as ctx:
                client.decide("hello", {"q": NoulQ("q")})
        self.assertIn("typesafe-sdk is not installed", str(ctx.exception))

    def test_missing_api_key_raises_decision_error(self):
        import os

        os.environ.pop("TYPESAFE_API_KEY", None)
        sys.modules["typesafe_sdk"] = _make_fake_sdk_module()

        from decisions.jev_client import JevDecisionClient

        client = JevDecisionClient({"model_name": "jev-latest"}, "jev")
        with self.assertRaises(DecisionError) as ctx:
            client.decide("hello", {"q": NoulQ("q")})
        self.assertIn("TYPESAFE_API_KEY", str(ctx.exception))

    def test_answer_mapping_against_fake_sdk(self):
        sys.modules["typesafe_sdk"] = _make_fake_sdk_module()

        from decisions.jev_client import JevDecisionClient

        client = JevDecisionClient({"api_key": "fake-key", "model_name": "jev-latest"}, "jev")
        response = client.decide(
            "state text",
            {
                "flag": NoulQ("is it true?"),
                "intent": ChoiceQ("what intent?", {"a": "desc a", "b": "desc b"}),
                "quality": ScoreQ("rate it", [("low", "l"), ("mid", "m"), ("high", "h")]),
            },
        )
        self.assertEqual(response.answers["flag"].kind, "noul")
        self.assertEqual(response.answers["flag"].value, 0.7)
        self.assertAlmostEqual(response.answers["flag"].confidence, 0.7)

        self.assertEqual(response.answers["intent"].kind, "choice")
        self.assertEqual(response.answers["intent"].value, "a")
        self.assertEqual(response.answers["intent"].confidence, 0.9)

        self.assertEqual(response.answers["quality"].kind, "score")
        self.assertAlmostEqual(response.answers["quality"].value, 1.0)  # index 2 of 3 -> normalized 1.0
        self.assertIn("high", response.answers["quality"].probabilities)
        self.assertEqual(response.est_input_tokens, 42)


def _make_fake_sdk_module() -> types.ModuleType:
    """A minimal stand-in for typesafe_sdk covering the request/response shape
    JevDecisionClient depends on, so the mapping logic is exercised without
    the real SDK or a network call.
    """
    mod = types.ModuleType("typesafe_sdk")

    class Noul:
        def __init__(self, instructions):
            self.instructions = instructions

    class Choice:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    class Score:
        def __init__(self, instructions, criteria):
            self.instructions = instructions
            self.criteria = criteria

    class _Answer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Usage:
        def __init__(self, input_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = None

    class _Response:
        def __init__(self, answers, model, usage):
            self.answers = answers
            self.model = model
            self.usage = usage

    class TypeSafeClient:
        def __init__(self, *, api_key=None, model=None, timeout=None):
            import os

            resolved_key = api_key or os.environ.get("TYPESAFE_API_KEY")
            if not resolved_key:
                raise RuntimeError(
                    "No API key was provided. Pass api_key or set the TYPESAFE_API_KEY environment variable."
                )
            self.api_key = resolved_key
            self.model = model
            self.timeout = timeout

        def system_one(self, *, state, questions):
            answers = {}
            for name, q in questions.items():
                if isinstance(q, Noul):
                    answers[name] = _Answer(type="noul", noul=0.7)
                elif isinstance(q, Choice):
                    labels = list(q.criteria.keys())
                    probs = {labels[0]: 0.9}
                    for l in labels[1:]:
                        probs[l] = 0.1 / max(1, len(labels) - 1)
                    answers[name] = _Answer(type="choice", choice=labels[0], confidence=0.9, probabilities=probs)
                elif isinstance(q, Score):
                    n = len(q.criteria)
                    top_index = n - 1
                    probs = {top_index: 1.0}
                    answers[name] = _Answer(
                        type="score", score=float(top_index), confidence=1.0, legend={}, probabilities=probs
                    )
            return _Response(answers=answers, model=self.model or "jev-latest", usage=_Usage(input_tokens=42))

    mod.Noul = Noul
    mod.Choice = Choice
    mod.Score = Score
    mod.TypeSafeClient = TypeSafeClient
    return mod


if __name__ == "__main__":
    unittest.main()
