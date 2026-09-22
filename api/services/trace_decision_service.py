"""Trace decision service — Jev-scored online trace risk tagging and the
bridge from a flagged trace into the HITL pending queue.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from api.services.trace_service import TraceStore
from decisions.config import build_decision_client
from decisions.traces import decide_trace, trace_io
from utils.human_annotations import AnnotationManager
from utils.logger import get_logger

logger = get_logger(__name__)


class TraceDecisionService:
    def __init__(
        self,
        store: TraceStore,
        config_path: str = "config/models.yaml",
        decision_client_factory: Optional[Callable[[str, str], Any]] = None,
        annotation_manager: Optional[AnnotationManager] = None,
    ) -> None:
        self._store = store
        self.config_path = config_path
        self.decision_client_factory = decision_client_factory or build_decision_client
        self._annotations = annotation_manager or AnnotationManager()
        self._decision_clients: Dict[str, Any] = {}

    def _get_client(self, model_key: str) -> Any:
        if model_key not in self._decision_clients:
            self._decision_clients[model_key] = self.decision_client_factory(model_key, self.config_path)
        return self._decision_clients[model_key]

    async def decide(self, trace_id: str, decision_model: str) -> Optional[Dict[str, Any]]:
        trace = await self._store.get(trace_id)
        if trace is None:
            return None
        client = self._get_client(decision_model)
        trace_dict = trace.model_dump()
        result = await asyncio.get_running_loop().run_in_executor(None, decide_trace, client, trace_dict)
        for tag_value in result["tags"]:
            await self._store.tag(trace_id, tag_value)
        await self._store.update_metadata(trace_id, "decisions", result["signals"])
        return result

    async def decide_batch(
        self,
        decision_model: str,
        *,
        trace_ids: Optional[List[str]] = None,
        tag: Optional[str] = None,
        limit: int = 200,
        force: bool = False,
    ) -> Dict[str, int]:
        if trace_ids is not None:
            fetched = [await self._store.get(tid) for tid in trace_ids]
            traces = [t for t in fetched if t is not None]
        else:
            traces = await self._store.list(tag=tag, limit=limit)

        decided = flagged = errors = 0
        for trace in traces:
            if not force and "decided:jev" in trace.tags:
                continue
            try:
                result = await self.decide(trace.trace_id, decision_model)
            except Exception as exc:  # noqa: BLE001 — one bad trace must not stop the batch
                logger.warning("[trace_decision] decide failed for %s: %s", trace.trace_id, exc)
                errors += 1
                continue
            if result is None:
                continue
            decided += 1
            if result["needs_review"]:
                flagged += 1
        return {"decided": decided, "flagged": flagged, "errors": errors}

    async def to_hitl(self, trace_id: str) -> Optional[str]:
        """Queue a flagged trace for human review. None if the trace doesn't
        exist or isn't flagged; idempotent (returns the existing item_id
        without re-adding if it's already queued)."""
        trace = await self._store.get(trace_id)
        if trace is None:
            return None
        if "needs_review" not in trace.tags:
            return None

        item_id = f"trace::{trace_id}"
        if any(item.get("item_id") == item_id for item in self._annotations.get_pending_items()):
            return item_id

        signals = trace.metadata.get("decisions") or {}
        risk_values = [v for k, v in signals.items() if k != "quality" and isinstance(v, (int, float))]
        max_risk = max(risk_values, default=0.0)
        user_input, assistant_output = trace_io(trace.model_dump())
        flagged_tags = sorted(t for t in trace.tags if t.startswith("risk:") or t == "quality:low")

        self._annotations.add_pending_item(
            {
                "item_id": item_id,
                "model_name": str(trace.metadata.get("model") or ""),
                "test_category": "online_trace",
                "test_id": trace_id,
                "question": user_input,
                "model_response": assistant_output,
                "llm_judge_score": round(1.0 - max_risk, 4),
                "judge_backend": "decision",
                # Points scale, not 0-1 — matches review_priority's convention
                # elsewhere (utils/human_annotations.LOW_CONFIDENCE_PRIORITY_FLOOR).
                "review_priority": round(max_risk * 100, 3),
                "queue_reason": f"Online trace flagged: {', '.join(flagged_tags)}",
                "metadata": {
                    "trace_id": trace_id,
                    "decisions": signals,
                    "tags": list(trace.tags),
                    "source": "online_trace",
                },
            }
        )
        return item_id
