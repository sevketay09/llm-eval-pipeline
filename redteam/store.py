"""
redteam/store.py — Red-team session dataclasses + in-memory store.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "persona_override",
    "boundary_test",
    "role_confusion",
    "tool_result_injection",
    "tool_poisoning",
]


@dataclass
class Attack:
    attack_id: str
    category: str
    name: str
    payload: str

    def to_dict(self) -> Dict:
        return {
            "attack_id": self.attack_id,
            "category": self.category,
            "name": self.name,
            "payload": self.payload,
        }


@dataclass
class AttackResult:
    attack_id: str
    category: str
    name: str
    payload: str
    response: str
    passed: bool
    reason: str
    latency_ms: float
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "attack_id": self.attack_id,
            "category": self.category,
            "name": self.name,
            "payload": self.payload,
            "response": self.response,
            "passed": self.passed,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class RedTeamSession:
    session_id: str
    system_prompt: str
    categories: List[str]
    attacks: List[Attack] = field(default_factory=list)
    results: List[AttackResult] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | error
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "categories": self.categories,
            "attacks": [a.to_dict() for a in self.attacks],
            "results": [r.to_dict() for r in self.results],
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_MAX_SESSIONS = 200


class RedTeamStore:
    def __init__(self) -> None:
        self._store: Dict[str, RedTeamSession] = {}
        self._order: List[str] = []

    def create(self, session: RedTeamSession) -> RedTeamSession:
        if session.session_id not in self._store:
            self._order.append(session.session_id)
        self._store[session.session_id] = session
        while len(self._order) > _MAX_SESSIONS:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        return session

    def get(self, session_id: str) -> Optional[RedTeamSession]:
        return self._store.get(session_id)

    def list(self, limit: int = 50) -> List[RedTeamSession]:
        return [self._store[sid] for sid in self._order[-limit:] if sid in self._store]

    def update(self, session: RedTeamSession) -> None:
        self._store[session.session_id] = session

    def count(self) -> int:
        return len(self._store)


def make_session(
    system_prompt: str,
    categories: List[str],
    session_id: Optional[str] = None,
) -> RedTeamSession:
    return RedTeamSession(
        session_id=session_id or uuid.uuid4().hex,
        system_prompt=system_prompt,
        categories=categories,
    )
