"""
redteam/runner.py — Runs attacks against a model_fn and scores responses.
Injectable model_fn and score_fn — no real LLM needed in tests.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from redteam.scorer import score_response as _default_score
from redteam.store import Attack, AttackResult, RedTeamSession

ModelFn = Callable[[str, str], Tuple[str, float]]
"""(system_prompt, user_input) -> (response_text, latency_ms)"""

ScoreFn = Callable[[Attack, str], Tuple[str, bool]]
"""(attack, response) -> (reason, passed)"""


class RedTeamRunner:
    def __init__(
        self,
        model_fn: ModelFn,
        score_fn: Optional[ScoreFn] = None,
    ) -> None:
        self._model_fn = model_fn
        self._score_fn = score_fn or _default_score

    def run_session(self, session: RedTeamSession) -> List[AttackResult]:
        results: List[AttackResult] = []
        for attack in session.attacks:
            try:
                response, latency_ms = self._model_fn(session.system_prompt, attack.payload)
                reason, passed = self._score_fn(attack, response)
                results.append(
                    AttackResult(
                        attack_id=attack.attack_id,
                        category=attack.category,
                        name=attack.name,
                        payload=attack.payload,
                        response=response,
                        passed=passed,
                        reason=reason,
                        latency_ms=round(latency_ms, 2),
                    )
                )
            except Exception as exc:
                results.append(
                    AttackResult(
                        attack_id=attack.attack_id,
                        category=attack.category,
                        name=attack.name,
                        payload=attack.payload,
                        response="",
                        passed=False,
                        reason="",
                        latency_ms=0.0,
                        error=str(exc),
                    )
                )
        return results
