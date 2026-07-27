"""Red-team service — orchestrates session generation and attack runs."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from api.schemas.redteam import AttackResultSchema, AttackSchema, SessionDetail, SessionSummary
from redteam.generator import generate_attacks
from redteam.runner import RedTeamRunner
from redteam.store import AttackResult, RedTeamSession, RedTeamStore, make_session


def _noop_model_fn(system_prompt: str, user_input: str) -> Tuple[str, float]:
    """Placeholder — used only when the session has no model_key set."""
    return f"[no model] prompt={system_prompt[:20]!r} input={user_input[:20]!r}", 0.0


def _default_adapter_factory(model_key: str, config_path: str) -> Any:
    """Build a UnifiedLLMAdapter for `model_key` with ${ENV_VAR} expansion.

    Mirrors SkillEvalService/CustomMetricService/ExperimentService's factory —
    ConfigService's loader does not expand ${ENV_VAR} placeholders, so reading
    straight from it would hand the adapter a literal "${OPENROUTER_API_KEY}".
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    config = yaml.safe_load(config_str)
    if model_key not in config.get("models", {}):
        raise ValueError(f"Model '{model_key}' not found in config")
    from adapters.unified_adapter import UnifiedLLMAdapter  # heavy import kept lazy

    return UnifiedLLMAdapter(dict(config["models"][model_key]), model_key=model_key)


class RedTeamService:
    def __init__(
        self,
        store: Optional[RedTeamStore] = None,
        config_path: str = "config/models.yaml",
        adapter_factory: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        self._store = store or RedTeamStore()
        self._lock = asyncio.Lock()
        self.config_path = config_path
        self.adapter_factory = adapter_factory or _default_adapter_factory
        self._adapters: Dict[str, Any] = {}

    def _get_adapter(self, model_key: str) -> Any:
        if model_key not in self._adapters:
            self._adapters[model_key] = self.adapter_factory(model_key, self.config_path)
        return self._adapters[model_key]

    def _build_model_fn(self, model_key: str):
        """Wrap UnifiedLLMAdapter.generate() into the (system_prompt, user_input)
        -> (output, latency_ms) shape RedTeamRunner expects. Raises on a
        failed generation instead of returning empty text, so the runner's
        existing per-attack exception handling records it as a real failure
        rather than a fabricated 'passed'."""
        adapter = self._get_adapter(model_key)

        def model_fn(system_prompt: str, user_input: str) -> Tuple[str, float]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            result = adapter.generate(messages)
            if result.get("error") or result.get("content") is None:
                raise RuntimeError(result.get("error") or "Model returned no content")
            return result["content"], round(result.get("latency", 0.0) * 1000, 2)

        return model_fn

    def create(self, system_prompt: str, categories: List[str], model_key: str = "") -> RedTeamSession:
        session = make_session(system_prompt=system_prompt, categories=categories, model_key=model_key)
        session.attacks = generate_attacks(system_prompt, categories)
        self._store.create(session)
        return session

    def get(self, session_id: str) -> Optional[RedTeamSession]:
        return self._store.get(session_id)

    def list(self, limit: int = 50) -> List[RedTeamSession]:
        return self._store.list(limit=limit)

    def save_state(self, path: Path) -> None:
        self._store.save(path)

    def load_state(self, path: Path) -> None:
        self._store.load_from(path)

    async def run(
        self,
        session_id: str,
        model_fn=None,
    ) -> Optional[RedTeamSession]:
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            if session.status == "running":
                return session
            session.status = "running"
            self._store.update(session)

        try:
            fn = model_fn or (self._build_model_fn(session.model_key) if session.model_key else _noop_model_fn)
            runner = RedTeamRunner(model_fn=fn)
            results: List[AttackResult] = await asyncio.get_event_loop().run_in_executor(
                None, runner.run_session, session
            )
            session.results = results
            session.status = "done"
        except Exception as exc:
            session.status = "error"
            session.error = str(exc)
        finally:
            session.finished_at = time.time()
            self._store.update(session)

        return session

    def to_summary(self, session: RedTeamSession) -> SessionSummary:
        passed = sum(1 for r in session.results if r.passed)
        failed = sum(1 for r in session.results if not r.passed)
        return SessionSummary(
            session_id=session.session_id,
            system_prompt=session.system_prompt,
            categories=session.categories,
            model_key=session.model_key,
            attack_count=len(session.attacks),
            status=session.status,
            passed=passed,
            failed=failed,
            created_at=session.created_at,
            finished_at=session.finished_at,
        )

    def to_detail(self, session: RedTeamSession) -> SessionDetail:
        passed = sum(1 for r in session.results if r.passed)
        failed = sum(1 for r in session.results if not r.passed)
        return SessionDetail(
            session_id=session.session_id,
            system_prompt=session.system_prompt,
            categories=session.categories,
            model_key=session.model_key,
            attacks=[AttackSchema(**a.to_dict()) for a in session.attacks],
            results=[AttackResultSchema(**r.to_dict()) for r in session.results],
            status=session.status,
            error=session.error,
            passed=passed,
            failed=failed,
            created_at=session.created_at,
            finished_at=session.finished_at,
        )
