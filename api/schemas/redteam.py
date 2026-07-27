"""Pydantic schemas for red-team endpoints."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from redteam.store import CATEGORIES


class CreateSessionRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1)
    categories: List[str] = Field(default_factory=lambda: list(CATEGORIES))


class AttackSchema(BaseModel):
    attack_id: str
    category: str
    name: str
    payload: str


class AttackResultSchema(BaseModel):
    attack_id: str
    category: str
    name: str
    payload: str
    response: str
    passed: bool
    reason: str
    latency_ms: float
    error: str = ""


class SessionSummary(BaseModel):
    session_id: str
    system_prompt: str
    categories: List[str]
    attack_count: int
    status: str
    passed: int = 0
    failed: int = 0
    created_at: float
    finished_at: Optional[float] = None


class SessionDetail(BaseModel):
    session_id: str
    system_prompt: str
    categories: List[str]
    attacks: List[AttackSchema]
    results: List[AttackResultSchema]
    status: str
    error: str = ""
    passed: int = 0
    failed: int = 0
    created_at: float
    finished_at: Optional[float] = None
