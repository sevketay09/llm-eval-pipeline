"""Shared decision-question catalogs, one place per domain. Populated one
madde at a time as each is implemented (redteam, safety/quality/agent judges,
skill routing, trace risk, failure taxonomy, RAG, dataset QC, conversation).
"""
from __future__ import annotations

from typing import List, Tuple

FIVE_LEVELS: List[Tuple[str, str]] = [
    ("1", "very poor"),
    ("2", "poor"),
    ("3", "acceptable"),
    ("4", "good"),
    ("5", "excellent"),
]
