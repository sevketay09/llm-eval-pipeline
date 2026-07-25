"""Red-team service — orchestrates session generation and attack runs."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple

from api.schemas.redteam import AttackResultSchema, AttackSchema, SessionDetail, SessionSummary
from redteam.generator import generate_attacks
from redteam.runner import RedTeamRunner
from redteam.store import AttackResult, RedTeamSession, RedTeamStore, make_session


def _noop_model_fn(system_prompt: str, user_input: str) -> Tuple[str, float]:
    return f"[no model] prompt={system_prompt[:20]!r} input={user_input[:20]!r}", 0.0


class RedTeamService:
    def __init__(self, store: Optional[RedTeamStore] = None) -> None:
        self._store = store or RedTeamStore()
        self._lock = asyncio.Lock()

    def create(self, system_prompt: str, categories: List[str]) -> RedTeamSession:
        session = make_session(system_prompt=system_prompt, categories=categories)
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

        fn = model_fn or _noop_model_fn
        runner = RedTeamRunner(model_fn=fn)
        try:
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
            attacks=[AttackSchema(**a.to_dict()) for a in session.attacks],
            results=[AttackResultSchema(**r.to_dict()) for r in session.results],
            status=session.status,
            error=session.error,
            passed=passed,
            failed=failed,
            created_at=session.created_at,
            finished_at=session.finished_at,
        )
