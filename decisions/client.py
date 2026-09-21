"""Decision client protocol, shared stats, and the retry/truncation base class."""
from __future__ import annotations

import statistics
import threading
import time
from typing import Dict, List, Optional, Protocol

from decisions.types import Decision, DecisionError, DecisionResponse, Question

DEFAULT_MAX_STATE_CHARS = 60_000
DEFAULT_MAX_RETRIES = 2
DEFAULT_COST_PER_MTOK_INPUT = 0.042


class DecisionClient(Protocol):
    model_key: str
    model_name: str
    stats: "DecisionStats"

    def decide(self, state: str, questions: Dict[str, Question]) -> DecisionResponse: ...


class DecisionStats:
    """Thread-safe running stats for a decision client. Never carries the API key."""

    def __init__(self, cost_per_mtok_input: float = DEFAULT_COST_PER_MTOK_INPUT) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.escalations = 0
        self.latencies_ms: List[float] = []
        self.est_input_tokens = 0
        self.cost_per_mtok_input = cost_per_mtok_input

    def record(self, response: DecisionResponse) -> None:
        with self._lock:
            self.requests += 1
            self.latencies_ms.append(response.latency_ms)
            self.est_input_tokens += response.est_input_tokens

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_escalation(self) -> None:
        with self._lock:
            self.escalations += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.latencies_ms)
            return {
                "requests": self.requests,
                "errors": self.errors,
                "escalations": self.escalations,
                "latency_p50_ms": _percentile(latencies, 50),
                "latency_p95_ms": _percentile(latencies, 95),
                "est_input_tokens": self.est_input_tokens,
                "est_cost_usd": round(self.est_input_tokens / 1_000_000 * self.cost_per_mtok_input, 6),
            }


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]


class BaseDecisionClient:
    """Shared truncation, retry, and stats bookkeeping for decision backends.

    Subclasses implement `_decide_raw`. Set `handles_own_retries = True` when
    the backend already retries transient failures internally (e.g. an SDK
    with its own retry policy) so this base class makes a single attempt
    instead of retrying on top of it.
    """

    handles_own_retries = False

    def __init__(
        self,
        model_key: str,
        model_name: str,
        *,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cost_per_mtok_input: float = DEFAULT_COST_PER_MTOK_INPUT,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self._max_state_chars = max_state_chars
        self._max_retries = max_retries
        self.stats = DecisionStats(cost_per_mtok_input=cost_per_mtok_input)

    def _decide_raw(self, state: str, questions: Dict[str, Question]) -> DecisionResponse:
        raise NotImplementedError

    def _truncate(self, state: str) -> str:
        if len(state) <= self._max_state_chars:
            return state
        return state[: self._max_state_chars] + "\n[...truncated]"

    def decide(self, state: str, questions: Dict[str, Question]) -> DecisionResponse:
        truncated = self._truncate(state)
        attempts = 1 if self.handles_own_retries else self._max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                response = self._decide_raw(truncated, questions)
                self.stats.record(response)
                return response
            except Exception as exc:  # noqa: BLE001 — any backend failure is a decision failure
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(2**attempt)

        self.stats.record_error()
        if isinstance(last_error, DecisionError):
            raise last_error
        raise DecisionError(str(last_error)) from last_error


__all__ = ["DecisionClient", "DecisionStats", "BaseDecisionClient", "Decision"]
