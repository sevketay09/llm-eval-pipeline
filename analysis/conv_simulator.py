"""
Conversation Simulator — G9

Simulates multi-turn conversations between a synthetic user (persona) and
an agent, then evaluates the full trajectory for goal completion, coherence,
and efficiency.

Standalone — no api/, utils/, adapters/ imports.
Injectable callables: agent_fn(messages) -> str, user_fn(messages, persona, turn) -> str.

CLI:
    python -m analysis.conv_simulator --persona persona.json --output result.json
    python -m analysis.conv_simulator --demo
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Persona:
    name: str
    goal: str
    style: str = "neutral"          # neutral | friendly | demanding | confused
    constraints: List[str] = field(default_factory=list)
    opening_message: Optional[str] = None
    max_turns: int = 10
    goal_keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.goal_keywords:
            self.goal_keywords = _extract_keywords(self.goal)


@dataclass
class Turn:
    turn_number: int
    user_message: str
    agent_response: str
    goal_coverage: float = 0.0     # fraction of goal_keywords covered so far


@dataclass
class Trajectory:
    persona: Dict[str, Any]
    turns: List[Turn]
    terminated_early: bool = False
    termination_reason: str = ""

    @property
    def all_agent_text(self) -> str:
        return " ".join(t.agent_response for t in self.turns).lower()


@dataclass
class TrajectoryResult:
    persona_name: str
    goal: str
    total_turns: int
    goal_completion_score: float   # 0–1: keyword coverage in agent responses
    trajectory_coherence: float    # 0–1: avg Jaccard between consecutive turns
    avg_response_relevance: float  # 0–1: per-turn relevance to goal
    efficiency_score: float        # goal_completion / normalised turn cost
    goal_achieved: bool            # completion >= 0.6
    turn_by_turn: List[Dict[str, Any]]
    summary: str


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "on", "at", "by", "for", "with", "from", "that", "this", "and",
    "or", "but", "not", "as", "it", "its", "i", "we", "you", "he", "she",
    "they", "my", "our", "your", "their", "how", "what", "when", "where",
    "why", "who", "which", "bir", "ve", "da", "de", "ile",
    "bu", "için", "ne", "mi", "mı", "mu", "mü", "ya", "ki",
}


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-ZçğışöüÇĞİŞÖÜ]{3,}\b", text.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOPWORDS))


def _keyword_coverage(keywords: List[str], text: str) -> float:
    if not keywords:
        return 1.0
    text_lower = text.lower()
    covered = sum(1 for kw in keywords if kw in text_lower)
    return covered / len(keywords)


def _jaccard(a: str, b: str) -> float:
    set_a = set(re.findall(r"\b\w+\b", a.lower()))
    set_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Rule-based user simulator
# ---------------------------------------------------------------------------

_STYLE_TEMPLATES: Dict[str, List[str]] = {
    "neutral": [
        "{goal}",
        "Can you help me with: {goal}?",
        "I need more details about {goal}.",
        "Please elaborate on {missing}.",
        "That's helpful. What about {missing}?",
        "I still need to know about {missing}.",
        "Could you be more specific about {missing}?",
        "Thank you, but I have another question about {missing}.",
    ],
    "friendly": [
        "Hi! I was hoping you could help me with {goal}.",
        "Thanks! Could you tell me more about {missing}?",
        "That's great, but I'm also curious about {missing}.",
        "Wonderful! One more thing — {missing}?",
        "You're very helpful! Can you also cover {missing}?",
    ],
    "demanding": [
        "I need {goal}. Now.",
        "You didn't answer about {missing}. Address it.",
        "Still missing {missing}. Be more thorough.",
        "Not good enough. Cover {missing} specifically.",
        "I require complete information on {missing}.",
    ],
    "confused": [
        "I'm not sure, but I think I need {goal}?",
        "Hmm, I don't quite understand. Can you explain {missing} differently?",
        "Wait, I'm confused about {missing}. Can you clarify?",
        "I'm still not clear on {missing}. Could you try again?",
        "Sorry, what did you mean about {missing}?",
    ],
}


def _default_user_fn(
    messages: List[Dict[str, str]],
    persona: Persona,
    turn_number: int,
) -> str:
    templates = _STYLE_TEMPLATES.get(persona.style, _STYLE_TEMPLATES["neutral"])

    if turn_number == 0:
        if persona.opening_message:
            return persona.opening_message
        t = templates[0] if templates else "{goal}"
        return t.format(goal=persona.goal, missing=persona.goal)

    # Find which goal keywords are still missing from agent responses so far
    agent_text = " ".join(
        m["content"] for m in messages if m.get("role") == "assistant"
    ).lower()
    missing_kws = [
        kw for kw in persona.goal_keywords if kw not in agent_text
    ]
    missing_phrase = (
        ", ".join(missing_kws[:3]) if missing_kws else persona.goal
    )

    # Cycle through follow-up templates
    idx = min(turn_number, len(templates) - 1)
    return templates[idx].format(goal=persona.goal, missing=missing_phrase)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_conversation(
    agent_fn: Callable[[List[Dict[str, str]]], str],
    persona: Persona,
    user_fn: Optional[Callable[[List[Dict[str, str]], Persona, int], str]] = None,
    max_turns: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Trajectory:
    """Run a simulated conversation and return the full trajectory."""
    _user_fn = user_fn if user_fn is not None else _default_user_fn
    max_t = max_turns if max_turns is not None else persona.max_turns

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns: List[Turn] = []
    accumulated_agent_text = ""

    for t in range(max_t):
        user_msg = _user_fn(messages, persona, t)
        messages.append({"role": "user", "content": user_msg})

        try:
            agent_resp = agent_fn(messages)
        except Exception as exc:  # noqa: BLE001
            return Trajectory(
                persona=asdict(persona),
                turns=turns,
                terminated_early=True,
                termination_reason=f"agent_fn raised: {exc}",
            )

        messages.append({"role": "assistant", "content": agent_resp})
        accumulated_agent_text += " " + agent_resp

        coverage = _keyword_coverage(persona.goal_keywords, accumulated_agent_text)
        turns.append(Turn(
            turn_number=t + 1,
            user_message=user_msg,
            agent_response=agent_resp,
            goal_coverage=round(coverage, 4),
        ))

        # Early termination: goal fully covered
        if coverage >= 1.0 and t >= 1:
            return Trajectory(
                persona=asdict(persona),
                turns=turns,
                terminated_early=True,
                termination_reason="goal_fully_covered",
            )

    return Trajectory(persona=asdict(persona), turns=turns)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_trajectory(
    trajectory: Trajectory,
    goal_eval_fn: Optional[Callable[[str, str], float]] = None,
) -> TrajectoryResult:
    """
    Evaluate a trajectory.

    goal_eval_fn(goal, full_agent_text) -> float (0–1) overrides keyword scoring.
    """
    persona_data = trajectory.persona
    persona_name = persona_data.get("name", "unknown")
    goal = persona_data.get("goal", "")
    goal_keywords: List[str] = persona_data.get("goal_keywords") or _extract_keywords(goal)
    n_turns = len(trajectory.turns)

    if n_turns == 0:
        return TrajectoryResult(
            persona_name=persona_name,
            goal=goal,
            total_turns=0,
            goal_completion_score=0.0,
            trajectory_coherence=0.0,
            avg_response_relevance=0.0,
            efficiency_score=0.0,
            goal_achieved=False,
            turn_by_turn=[],
            summary="No turns recorded.",
        )

    all_agent = trajectory.all_agent_text

    # 1. Goal completion
    if goal_eval_fn is not None:
        goal_completion = float(goal_eval_fn(goal, all_agent))
    else:
        goal_completion = _keyword_coverage(goal_keywords, all_agent)

    # 2. Trajectory coherence — avg Jaccard between consecutive turn texts
    coherence_scores: List[float] = []
    for i in range(1, n_turns):
        prev = (
            trajectory.turns[i - 1].user_message
            + " "
            + trajectory.turns[i - 1].agent_response
        )
        curr = (
            trajectory.turns[i].user_message
            + " "
            + trajectory.turns[i].agent_response
        )
        coherence_scores.append(_jaccard(prev, curr))
    trajectory_coherence = (
        sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
    )

    # 3. Per-turn response relevance to goal
    relevance_scores: List[float] = []
    for turn in trajectory.turns:
        rel = _keyword_coverage(goal_keywords, turn.agent_response)
        relevance_scores.append(rel)
    avg_relevance = sum(relevance_scores) / len(relevance_scores)

    # 4. Efficiency — goal completion adjusted for turns used
    # max_turns from persona
    max_turns = persona_data.get("max_turns", 10) or 10
    turn_penalty = math.log1p(n_turns) / math.log1p(max_turns)
    efficiency = goal_completion * (1.0 - 0.3 * turn_penalty)
    efficiency = max(0.0, min(1.0, efficiency))

    goal_achieved = goal_completion >= 0.6

    # Build per-turn details
    turn_by_turn: List[Dict[str, Any]] = []
    for i, turn in enumerate(trajectory.turns):
        turn_by_turn.append({
            "turn": turn.turn_number,
            "user_message": turn.user_message,
            "agent_response": turn.agent_response,
            "goal_coverage": turn.goal_coverage,
            "response_relevance": round(relevance_scores[i], 4),
        })

    # Summary
    status = "ACHIEVED" if goal_achieved else "NOT ACHIEVED"
    summary = (
        f"Goal {status} in {n_turns} turns. "
        f"Completion: {goal_completion:.0%}, "
        f"Coherence: {trajectory_coherence:.0%}, "
        f"Efficiency: {efficiency:.0%}."
    )
    if trajectory.terminated_early:
        summary += f" (Early stop: {trajectory.termination_reason})"

    return TrajectoryResult(
        persona_name=persona_name,
        goal=goal,
        total_turns=n_turns,
        goal_completion_score=round(goal_completion, 4),
        trajectory_coherence=round(trajectory_coherence, 4),
        avg_response_relevance=round(avg_relevance, 4),
        efficiency_score=round(efficiency, 4),
        goal_achieved=goal_achieved,
        turn_by_turn=turn_by_turn,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_simulation_suite(
    personas: List[Persona],
    agent_fn: Callable[[List[Dict[str, str]]], str],
    user_fn: Optional[Callable] = None,
    system_prompt: Optional[str] = None,
    goal_eval_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run all personas and return aggregated report."""
    results: List[Dict[str, Any]] = []
    for persona in personas:
        traj = simulate_conversation(
            agent_fn, persona, user_fn=user_fn, system_prompt=system_prompt
        )
        result = evaluate_trajectory(traj, goal_eval_fn=goal_eval_fn)
        results.append(asdict(result))

    if not results:
        return {"personas": [], "aggregate": {}}

    n = len(results)
    agg = {
        "total_personas": n,
        "goals_achieved": sum(1 for r in results if r["goal_achieved"]),
        "goal_achievement_rate": round(
            sum(1 for r in results if r["goal_achieved"]) / n, 4
        ),
        "avg_goal_completion": round(
            sum(r["goal_completion_score"] for r in results) / n, 4
        ),
        "avg_coherence": round(
            sum(r["trajectory_coherence"] for r in results) / n, 4
        ),
        "avg_turns": round(sum(r["total_turns"] for r in results) / n, 2),
        "avg_efficiency": round(
            sum(r["efficiency_score"] for r in results) / n, 4
        ),
    }
    return {"personas": results, "aggregate": agg}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_text(result: TrajectoryResult) -> str:
    lines = [
        f"Persona: {result.persona_name}",
        f"Goal: {result.goal}",
        f"Turns: {result.total_turns}",
        f"Goal Completion: {result.goal_completion_score:.0%}",
        f"Coherence: {result.trajectory_coherence:.0%}",
        f"Avg Relevance: {result.avg_response_relevance:.0%}",
        f"Efficiency: {result.efficiency_score:.0%}",
        f"Achieved: {'YES' if result.goal_achieved else 'NO'}",
        "",
        result.summary,
        "",
        "--- Turn-by-turn ---",
    ]
    for t in result.turn_by_turn:
        lines.append(
            f"  [{t['turn']}] User: {t['user_message'][:80]}"
        )
        lines.append(
            f"       Agent: {t['agent_response'][:80]} "
            f"(cov={t['goal_coverage']:.0%})"
        )
    return "\n".join(lines)


def _format_markdown(result: TrajectoryResult) -> str:
    achieved = "✅" if result.goal_achieved else "❌"
    lines = [
        f"## Conversation Simulation — {result.persona_name} {achieved}",
        "",
        f"**Goal:** {result.goal}",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Goal Completion | {result.goal_completion_score:.0%} |",
        f"| Coherence | {result.trajectory_coherence:.0%} |",
        f"| Avg Relevance | {result.avg_response_relevance:.0%} |",
        f"| Efficiency | {result.efficiency_score:.0%} |",
        f"| Turns | {result.total_turns} |",
        "",
        f"**Summary:** {result.summary}",
        "",
        "### Turn-by-turn",
    ]
    for t in result.turn_by_turn:
        lines.append(
            f"- **Turn {t['turn']}** (cov={t['goal_coverage']:.0%})"
        )
        lines.append(f"  - User: _{t['user_message'][:100]}_")
        lines.append(f"  - Agent: {t['agent_response'][:100]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo agent (echo-based for testing without LLM)
# ---------------------------------------------------------------------------

def _demo_agent_fn(messages: List[Dict[str, str]]) -> str:
    """Echo-based demo: reflects user keywords back in a templated response."""
    user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    keywords = _extract_keywords(user_msg)[:4]
    kw_str = ", ".join(keywords) if keywords else "your question"
    return (
        f"I can help you with {kw_str}. "
        f"Here is detailed information about {kw_str}: "
        f"This topic covers important aspects including {kw_str} and related areas."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.conv_simulator",
        description="Conversation Simulator — synthetic user persona evaluation",
    )
    p.add_argument(
        "--persona", metavar="FILE",
        help="JSON file with persona definition or list of personas",
    )
    p.add_argument("--output", metavar="FILE", help="Write result JSON to file")
    p.add_argument(
        "--format", choices=["text", "json", "markdown"], default="text",
    )
    p.add_argument(
        "--max-turns", type=int, default=None,
        help="Override persona max_turns",
    )
    p.add_argument("--demo", action="store_true", help="Run demo with built-in persona")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo:
        personas = [
            Persona(
                name="TechUser",
                goal="understand how to implement authentication in a REST API with JWT tokens",
                style="friendly",
                max_turns=5,
            ),
            Persona(
                name="DataSciUser",
                goal="learn about gradient descent optimization for neural network training",
                style="neutral",
                max_turns=5,
            ),
        ]
        report = run_simulation_suite(personas, _demo_agent_fn)
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            for r_dict in report["personas"]:
                r = TrajectoryResult(**r_dict)
                if args.format == "markdown":
                    print(_format_markdown(r))
                else:
                    print(_format_text(r))
                print()
            agg = report["aggregate"]
            print(
                f"AGGREGATE: {agg['goals_achieved']}/{agg['total_personas']} achieved, "
                f"avg_completion={agg['avg_goal_completion']:.0%}, "
                f"avg_turns={agg['avg_turns']:.1f}"
            )
        return 0

    if not args.persona:
        parser.print_help()
        return 2

    with open(args.persona, encoding="utf-8") as f:
        raw = json.load(f)

    persona_list = raw if isinstance(raw, list) else [raw]
    personas = [Persona(**p) for p in persona_list]

    report = run_simulation_suite(
        personas,
        _demo_agent_fn,
        system_prompt=None,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output}")
        return 0

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for r_dict in report["personas"]:
            r = TrajectoryResult(**r_dict)
            if args.format == "markdown":
                print(_format_markdown(r))
            else:
                print(_format_text(r))
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
