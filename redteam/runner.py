"""
redteam/runner.py — Runs attacks against a model_fn and scores responses.
Injectable model_fn and score_fn — no real LLM needed in tests.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from redteam.scorer import score_response as _default_score
from redteam.store import Attack, AttackResult, RedTeamSession

ModelFn = Callable[[str, str], Tuple[str, float]]
"""(system_prompt, user_input) -> (response_text, latency_ms)"""

ScoreFn = Callable[[Attack, str], Union[Tuple[str, bool], Tuple[str, bool, Dict[str, Any]]]]
"""(attack, response) -> (reason, passed) or (reason, passed, extra).

`extra` (decisions.redteam.make_decision_score_fn's shape) may carry:
"signals": Dict[str, float], "needs_review": bool, "scorer": str.
"""


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
                outcome = self._score_fn(attack, response)
                reason, passed = outcome[0], outcome[1]
                extra: Dict[str, Any] = outcome[2] if len(outcome) > 2 else {}
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
                        signals=extra.get("signals", {}),
                        needs_review=extra.get("needs_review", False),
                        scorer=extra.get("scorer", "heuristic"),
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
