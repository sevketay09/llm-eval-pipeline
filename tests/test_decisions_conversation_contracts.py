"""Contract tests for decisions/conversation.py and its wiring into
analysis/conv_simulator.py (madde 3.9). Offline only — MockDecisionClient,
no network.
"""
from __future__ import annotations

import unittest

from analysis.conv_simulator import (
    Persona,
    evaluate_trajectory,
    run_simulation_suite,
    simulate_conversation,
)
from decisions.conversation import make_stop_fn, make_trajectory_eval_fn, render_transcript
from decisions.mock_client import MockDecisionClient


def _persona(**overrides) -> Persona:
    defaults = dict(name="TestUser", goal="reset my password", max_turns=4)
    defaults.update(overrides)
    return Persona(**defaults)


def _agent_fn(messages):
    return "Here is how you reset your password."


class RenderTranscriptContractTests(unittest.TestCase):
    def test_includes_goal_and_turns(self):
        persona = _persona()
        traj = simulate_conversation(_agent_fn, persona)
        text = render_transcript(traj)
        self.assertIn("GOAL: reset my password", text)
        self.assertIn("USER:", text)
        self.assertIn("AGENT:", text)

    def test_truncated_from_the_start_past_max_chars(self):
        persona = _persona()
        traj = simulate_conversation(_agent_fn, persona)
        traj.turns[0].agent_response = "x" * 20000
        text = render_transcript(traj)
        self.assertLessEqual(len(text), 12000)
        self.assertNotIn("GOAL:", text)


class MakeTrajectoryEvalFnContractTests(unittest.TestCase):
    def test_returns_expected_keys(self):
        client = MockDecisionClient()
        eval_fn = make_trajectory_eval_fn(client)
        traj = simulate_conversation(_agent_fn, _persona())
        result = eval_fn(traj)
        self.assertEqual(
            set(result.keys()),
            {"goal_completion", "coherence", "relevance", "frustration", "looped", "confidence"},
        )

    def test_values_come_from_client(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.77, "coherence": 0.5})
        eval_fn = make_trajectory_eval_fn(client)
        traj = simulate_conversation(_agent_fn, _persona())
        result = eval_fn(traj)
        self.assertEqual(result["goal_completion"], 0.77)
        self.assertEqual(result["coherence"], 0.5)


class MakeStopFnContractTests(unittest.TestCase):
    def test_stops_when_above_threshold(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.95})
        stop_fn = make_stop_fn(client, threshold=0.9)
        self.assertTrue(stop_fn([{"role": "user", "content": "hi"}], _persona()))

    def test_does_not_stop_below_threshold(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.5})
        stop_fn = make_stop_fn(client, threshold=0.9)
        self.assertFalse(stop_fn([{"role": "user", "content": "hi"}], _persona()))


class ConvSimulatorDecisionWiringContractTests(unittest.TestCase):
    def test_stop_fn_ends_conversation_after_first_turn(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.95})
        stop_fn = make_stop_fn(client, threshold=0.9)
        traj = simulate_conversation(_agent_fn, _persona(max_turns=10), stop_fn=stop_fn)
        self.assertEqual(len(traj.turns), 1)
        self.assertTrue(traj.terminated_early)
        self.assertEqual(traj.termination_reason, "goal_completed")

    def test_trajectory_eval_fn_fills_decision_signals(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.8, "coherence": 0.6, "relevance": 0.4})
        eval_fn = make_trajectory_eval_fn(client)
        traj = simulate_conversation(_agent_fn, _persona())
        result = evaluate_trajectory(traj, trajectory_eval_fn=eval_fn)
        self.assertEqual(result.goal_completion_score, 0.8)
        self.assertEqual(result.trajectory_coherence, 0.6)
        self.assertEqual(result.avg_response_relevance, 0.4)
        self.assertIsNotNone(result.decision_signals)
        self.assertEqual(result.decision_signals["goal_completion"], 0.8)

    def test_trajectory_eval_fn_wins_over_goal_eval_fn(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.8})
        eval_fn = make_trajectory_eval_fn(client)
        traj = simulate_conversation(_agent_fn, _persona())
        result = evaluate_trajectory(
            traj, goal_eval_fn=lambda goal, text: 0.1, trajectory_eval_fn=eval_fn
        )
        self.assertEqual(result.goal_completion_score, 0.8)

    def test_no_eval_fn_leaves_decision_signals_none(self):
        traj = simulate_conversation(_agent_fn, _persona())
        result = evaluate_trajectory(traj)
        self.assertIsNone(result.decision_signals)

    def test_run_simulation_suite_passes_through(self):
        client = MockDecisionClient(overrides={"goal_completed": 0.95})
        eval_fn = make_trajectory_eval_fn(client)
        stop_fn = make_stop_fn(client, threshold=0.9)
        report = run_simulation_suite(
            [_persona()], _agent_fn, trajectory_eval_fn=eval_fn, stop_fn=stop_fn
        )
        result = report["personas"][0]
        self.assertEqual(result["total_turns"], 1)
        self.assertIsNotNone(result["decision_signals"])


if __name__ == "__main__":
    unittest.main()
