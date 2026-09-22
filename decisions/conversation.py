"""Jev-backed Conversation Simulator evaluation — one trajectory-level call
in place of analysis/conv_simulator.py's keyword/Jaccard heuristics, plus an
early-stop check usable per turn.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from decisions.client import DecisionClient
from decisions.types import NoulQ, ScoreQ

_MAX_TRANSCRIPT_CHARS = 12000

_THREE_LEVELS = [
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
]

_GOAL_COMPLETED_Q = NoulQ(
    "By the end of TRANSCRIPT the agent fully achieved the user's GOAL."
)


def render_transcript(trajectory: Any) -> str:
    """Persona goal + turn-by-turn USER/AGENT lines, truncated from the start
    to the last `_MAX_TRANSCRIPT_CHARS` characters.
    """
    persona = trajectory.persona
    lines = [f"GOAL: {persona.get('goal', '')}"]
    for turn in trajectory.turns:
        lines.append(f"USER: {turn.user_message}")
        lines.append(f"AGENT: {turn.agent_response}")
    text = "\n".join(lines)
    if len(text) > _MAX_TRANSCRIPT_CHARS:
        text = text[-_MAX_TRANSCRIPT_CHARS:]
    return text


def make_trajectory_eval_fn(client: DecisionClient) -> Callable[[Any], Dict[str, Any]]:
    def evaluate(trajectory: Any) -> Dict[str, Any]:
        state = render_transcript(trajectory)
        questions: Dict[str, Any] = {
            "goal_completed": _GOAL_COMPLETED_Q,
            "coherence": ScoreQ(
                "Rate how coherent and consistent the agent is across turns.", _THREE_LEVELS
            ),
            "relevance": ScoreQ(
                "Rate how focused the agent's replies are on the user's GOAL.", _THREE_LEVELS
            ),
            "frustration": NoulQ(
                "The user shows frustration or repeats themselves because of the agent."
            ),
            "looped": NoulQ("The agent repeats itself or gets stuck in a loop."),
        }
        answers = client.decide(state, questions).answers
        confidences = [a.confidence for a in answers.values() if a.confidence is not None]
        return {
            "goal_completion": answers["goal_completed"].value,
            "coherence": answers["coherence"].value,
            "relevance": answers["relevance"].value,
            "frustration": answers["frustration"].value,
            "looped": answers["looped"].value,
            "confidence": min(confidences) if confidences else None,
        }

    return evaluate


def make_stop_fn(
    client: DecisionClient, threshold: float = 0.9
) -> Callable[[List[Dict[str, str]], Any], bool]:
    def stop(messages: List[Dict[str, str]], persona: Any) -> bool:
        goal = persona.get("goal", "") if isinstance(persona, dict) else getattr(persona, "goal", "")
        lines = [f"GOAL: {goal}"]
        for message in messages:
            role = "USER" if message.get("role") == "user" else "AGENT"
            lines.append(f"{role}: {message.get('content', '')}")
        state = "\n".join(lines)
        if len(state) > _MAX_TRANSCRIPT_CHARS:
            state = state[-_MAX_TRANSCRIPT_CHARS:]
        answers = client.decide(state, {"goal_completed": _GOAL_COMPLETED_Q}).answers
        return answers["goal_completed"].value >= threshold

    return stop
