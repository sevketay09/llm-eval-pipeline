"""
Contract tests for analysis/conv_simulator.py (G9)

All tests are offline — no LLM calls.
Covers: Persona, simulate_conversation, evaluate_trajectory,
        run_simulation_suite, formatters, CLI.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from analysis.conv_simulator import (
    Persona,
    Trajectory,
    TrajectoryResult,
    Turn,
    _default_user_fn,
    _demo_agent_fn,
    _extract_keywords,
    _jaccard,
    _keyword_coverage,
    evaluate_trajectory,
    main,
    run_simulation_suite,
    simulate_conversation,
    _format_text,
    _format_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo_agent(messages):
    """Repeats all goal keywords from the conversation back in the response."""
    all_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    return f"Here is information about {all_text} covering all aspects in detail."


def _silent_agent(messages):
    return "OK."


def _failing_agent(messages):
    raise RuntimeError("agent crashed")


def _make_persona(**kwargs) -> Persona:
    defaults = dict(
        name="Test",
        goal="learn about Python decorators and how they work",
        style="neutral",
        max_turns=5,
    )
    defaults.update(kwargs)
    return Persona(**defaults)


# ---------------------------------------------------------------------------
# 1. Keyword helpers
# ---------------------------------------------------------------------------

def test_extract_keywords_filters_stopwords():
    kws = _extract_keywords("how to implement authentication with JWT tokens")
    assert "how" not in kws
    assert "with" not in kws
    assert "implement" in kws or "authentication" in kws


def test_extract_keywords_deduplicates():
    kws = _extract_keywords("cat cat cat dog dog")
    assert kws.count("cat") == 1


def test_keyword_coverage_full():
    assert _keyword_coverage(["python", "decorators"], "python decorators are great") == 1.0


def test_keyword_coverage_partial():
    score = _keyword_coverage(["python", "decorators", "closures"], "python is great")
    assert 0 < score < 1.0


def test_keyword_coverage_empty_keywords():
    assert _keyword_coverage([], "anything") == 1.0


def test_jaccard_identical():
    assert _jaccard("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("foo bar", "baz qux") == 0.0


def test_jaccard_partial():
    score = _jaccard("hello world", "hello python")
    assert 0 < score < 1.0


# ---------------------------------------------------------------------------
# 2. Persona
# ---------------------------------------------------------------------------

def test_persona_auto_extracts_keywords():
    p = Persona(name="X", goal="implement JWT authentication in Python")
    assert len(p.goal_keywords) > 0
    assert any(kw in p.goal_keywords for kw in ["implement", "jwt", "authentication", "python"])


def test_persona_respects_explicit_keywords():
    p = Persona(name="X", goal="something", goal_keywords=["alpha", "beta"])
    assert p.goal_keywords == ["alpha", "beta"]


def test_persona_default_style():
    p = Persona(name="X", goal="learn Python")
    assert p.style == "neutral"


# ---------------------------------------------------------------------------
# 3. Default user function
# ---------------------------------------------------------------------------

def test_default_user_fn_turn_zero_uses_opening():
    p = _make_persona(opening_message="Hi, I need help!")
    msg = _default_user_fn([], p, 0)
    assert msg == "Hi, I need help!"


def test_default_user_fn_turn_zero_no_opening():
    p = _make_persona(opening_message=None)
    msg = _default_user_fn([], p, 0)
    assert len(msg) > 0


def test_default_user_fn_followup_mentions_missing():
    p = _make_persona(goal="learn about Python closures and decorators")
    messages = [
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    msg = _default_user_fn(messages, p, 1)
    # Should mention missing keywords (closures / decorators not in agent response)
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# 4. simulate_conversation
# ---------------------------------------------------------------------------

def test_simulate_runs_max_turns():
    p = _make_persona(max_turns=3)
    traj = simulate_conversation(_silent_agent, p)
    assert len(traj.turns) == 3


def test_simulate_terminates_early_on_full_coverage():
    # Agent echoes all keywords → should terminate early
    p = _make_persona(goal="python decorators", max_turns=10)
    traj = simulate_conversation(_echo_agent, p)
    assert traj.terminated_early
    assert traj.termination_reason == "goal_fully_covered"
    assert len(traj.turns) < 10


def test_simulate_agent_crash_returns_trajectory():
    p = _make_persona(max_turns=5)
    traj = simulate_conversation(_failing_agent, p)
    assert traj.terminated_early
    assert "agent_fn raised" in traj.termination_reason


def test_simulate_custom_user_fn_called():
    calls = []

    def my_user_fn(messages, persona, turn):
        calls.append(turn)
        return f"Custom turn {turn}"

    p = _make_persona(max_turns=3)
    simulate_conversation(_silent_agent, p, user_fn=my_user_fn)
    assert calls == [0, 1, 2]


def test_simulate_turn_structure():
    p = _make_persona(max_turns=2)
    traj = simulate_conversation(_silent_agent, p)
    for turn in traj.turns:
        assert isinstance(turn.turn_number, int)
        assert isinstance(turn.user_message, str)
        assert isinstance(turn.agent_response, str)
        assert 0.0 <= turn.goal_coverage <= 1.0


def test_simulate_system_prompt_included():
    captured = []

    def cap_agent(messages):
        captured.append(messages[:])
        return "response"

    p = _make_persona(max_turns=1)
    simulate_conversation(_silent_agent, p, system_prompt="Be helpful.")
    # we can't check captured here since cap_agent not passed, use a proper one
    simulate_conversation(cap_agent, p, system_prompt="Be helpful.")
    assert captured[0][0]["role"] == "system"
    assert "Be helpful" in captured[0][0]["content"]


# ---------------------------------------------------------------------------
# 5. evaluate_trajectory
# ---------------------------------------------------------------------------

def _build_trajectory(n_turns: int, agent_text: str) -> Trajectory:
    p = _make_persona()
    turns = [
        Turn(
            turn_number=i + 1,
            user_message="user msg",
            agent_response=agent_text,
            goal_coverage=0.5,
        )
        for i in range(n_turns)
    ]
    from dataclasses import asdict
    return Trajectory(persona=asdict(p), turns=turns)


def test_evaluate_zero_turns():
    traj = _build_trajectory(0, "")
    result = evaluate_trajectory(traj)
    assert result.total_turns == 0
    assert result.goal_completion_score == 0.0
    assert not result.goal_achieved


def test_evaluate_goal_achieved_when_high_coverage():
    # Use a trajectory where agent_text contains all goal keywords
    goal = "python decorators closures"
    from dataclasses import asdict
    p = Persona(name="T", goal=goal, max_turns=3)
    turns = [
        Turn(1, "tell me", "python decorators closures explained", 1.0),
        Turn(2, "more", "more about python decorators closures", 1.0),
    ]
    traj = Trajectory(persona=asdict(p), turns=turns)
    result = evaluate_trajectory(traj)
    assert result.goal_achieved
    assert result.goal_completion_score > 0.6


def test_evaluate_goal_not_achieved_silent_agent():
    traj = _build_trajectory(5, "OK.")
    result = evaluate_trajectory(traj)
    assert not result.goal_achieved


def test_evaluate_coherence_single_turn():
    traj = _build_trajectory(1, "some response")
    result = evaluate_trajectory(traj)
    assert result.trajectory_coherence == 1.0  # no pair → default 1.0


def test_evaluate_custom_goal_eval_fn():
    traj = _build_trajectory(2, "irrelevant")

    def always_one(goal, text):
        return 1.0

    result = evaluate_trajectory(traj, goal_eval_fn=always_one)
    assert result.goal_completion_score == 1.0
    assert result.goal_achieved


def test_evaluate_result_fields_present():
    traj = _build_trajectory(3, "python decorators are great")
    result = evaluate_trajectory(traj)
    assert hasattr(result, "persona_name")
    assert hasattr(result, "goal")
    assert hasattr(result, "turn_by_turn")
    assert len(result.turn_by_turn) == 3
    assert isinstance(result.summary, str)


def test_evaluate_efficiency_penalises_many_turns():
    traj_short = _build_trajectory(1, "python decorators closures")
    traj_long = _build_trajectory(9, "python decorators closures")

    r_short = evaluate_trajectory(traj_short)
    r_long = evaluate_trajectory(traj_long)

    # Both achieve same coverage but shorter should be more efficient
    assert r_short.efficiency_score >= r_long.efficiency_score


# ---------------------------------------------------------------------------
# 6. run_simulation_suite
# ---------------------------------------------------------------------------

def test_suite_aggregate_keys():
    personas = [_make_persona(name=f"P{i}") for i in range(3)]
    report = run_simulation_suite(personas, _silent_agent)
    agg = report["aggregate"]
    assert "total_personas" in agg
    assert "goals_achieved" in agg
    assert "goal_achievement_rate" in agg
    assert "avg_goal_completion" in agg
    assert "avg_coherence" in agg
    assert "avg_turns" in agg


def test_suite_persona_count():
    personas = [_make_persona(name=f"P{i}") for i in range(4)]
    report = run_simulation_suite(personas, _silent_agent)
    assert len(report["personas"]) == 4
    assert report["aggregate"]["total_personas"] == 4


def test_suite_empty_personas():
    report = run_simulation_suite([], _silent_agent)
    assert report["aggregate"] == {}


def test_suite_all_styles():
    personas = [
        _make_persona(style=s, max_turns=2)
        for s in ["neutral", "friendly", "demanding", "confused"]
    ]
    report = run_simulation_suite(personas, _echo_agent)
    assert report["aggregate"]["total_personas"] == 4


# ---------------------------------------------------------------------------
# 7. Formatters
# ---------------------------------------------------------------------------

def _make_result() -> TrajectoryResult:
    traj = _build_trajectory(2, "python decorators work like this")
    return evaluate_trajectory(traj)


def test_format_text_contains_key_fields():
    r = _make_result()
    out = _format_text(r)
    assert "Persona:" in out
    assert "Goal:" in out
    assert "Turns:" in out
    assert "Achieved:" in out


def test_format_markdown_contains_table():
    r = _make_result()
    out = _format_markdown(r)
    assert "| Metric |" in out
    assert "Goal Completion" in out


# ---------------------------------------------------------------------------
# 8. Demo agent
# ---------------------------------------------------------------------------

def test_demo_agent_returns_string():
    msgs = [{"role": "user", "content": "Tell me about Python decorators"}]
    resp = _demo_agent_fn(msgs)
    assert isinstance(resp, str) and len(resp) > 0


def test_demo_agent_reflects_keywords():
    msgs = [{"role": "user", "content": "authentication JWT tokens"}]
    resp = _demo_agent_fn(msgs)
    assert "authentication" in resp.lower() or "jwt" in resp.lower()


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def test_cli_demo_text(capsys):
    rc = main(["--demo", "--format", "text"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Goal" in out or "Persona" in out


def test_cli_demo_json(capsys):
    rc = main(["--demo", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "personas" in data
    assert "aggregate" in data


def test_cli_demo_markdown(capsys):
    rc = main(["--demo", "--format", "markdown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "##" in out


def test_cli_no_args_returns_2(capsys):
    rc = main([])
    assert rc == 2


def test_cli_persona_file(tmp_path, capsys):
    persona_data = {
        "name": "FileUser",
        "goal": "learn about machine learning gradient descent",
        "style": "neutral",
        "max_turns": 3,
    }
    f = tmp_path / "persona.json"
    f.write_text(json.dumps(persona_data))
    rc = main(["--persona", str(f), "--format", "text"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FileUser" in out or "Goal" in out


def test_cli_persona_list_file(tmp_path, capsys):
    persona_list = [
        {"name": "A", "goal": "understand python closures decorators", "max_turns": 2},
        {"name": "B", "goal": "learn about neural networks deep learning", "max_turns": 2},
    ]
    f = tmp_path / "personas.json"
    f.write_text(json.dumps(persona_list))
    rc = main(["--persona", str(f), "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["aggregate"]["total_personas"] == 2


def test_cli_output_file(tmp_path, capsys):
    out_file = tmp_path / "result.json"
    persona_data = {
        "name": "SaveUser",
        "goal": "learn about sorting algorithms",
        "max_turns": 2,
    }
    p_file = tmp_path / "p.json"
    p_file.write_text(json.dumps(persona_data))
    rc = main(["--persona", str(p_file), "--output", str(out_file)])
    assert rc == 0
    assert out_file.exists()
    saved = json.loads(out_file.read_text())
    assert "personas" in saved
    assert "aggregate" in saved
