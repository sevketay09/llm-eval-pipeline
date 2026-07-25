"""Custom metric service — generate judge prompts + evaluate cases."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from api.schemas.custom_metrics import (
    CaseEvalResult,
    EvaluateCaseRequest,
    EvaluateMetricResponse,
    MetricDetail,
    MetricSummary,
)
from evaluators.custom_metric import (
    _parse_judge_response,
    _render_prompt,
    generate_judge_prompt,
)

_MAX_METRICS = 200


class _MetricRecord:
    def __init__(self, metric_id: str, name: str, description: str, prompt: str):
        self.metric_id = metric_id
        self.name = name
        self.description = description
        self.prompt = prompt
        self.status = "ready"
        self.created_at = time.time()

    def to_dict(self) -> Dict:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "_MetricRecord":
        rec = cls(
            metric_id=data["metric_id"],
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
        )
        rec.status = data.get("status", "ready")
        rec.created_at = data.get("created_at", time.time())
        return rec


class CustomMetricService:
    def __init__(self) -> None:
        self._store: Dict[str, _MetricRecord] = {}
        self._order: List[str] = []

    def create(self, name: str, description: str) -> _MetricRecord:
        prompt = generate_judge_prompt(description)
        rec = _MetricRecord(
            metric_id=uuid.uuid4().hex,
            name=name,
            description=description,
            prompt=prompt,
        )
        self._order.append(rec.metric_id)
        self._store[rec.metric_id] = rec
        while len(self._order) > _MAX_METRICS:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)
        return rec

    def get(self, metric_id: str) -> Optional[_MetricRecord]:
        return self._store.get(metric_id)

    def list(self) -> List[_MetricRecord]:
        return [self._store[mid] for mid in self._order if mid in self._store]

    def save(self, path: Path) -> None:
        """Snapshot current state to disk (atomic write) so a process restart doesn't lose it."""
        data = [self._store[mid].to_dict() for mid in self._order if mid in self._store]
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)

    def load_from(self, path: Path) -> None:
        """Replace current state with a previously saved snapshot, if one exists."""
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        self._store = {}
        self._order = []
        for item in raw:
            rec = _MetricRecord.from_dict(item)
            self._order.append(rec.metric_id)
            self._store[rec.metric_id] = rec

    def evaluate(
        self,
        metric_id: str,
        cases: List[EvaluateCaseRequest],
        llm_fn=None,
    ) -> EvaluateMetricResponse:
        rec = self._store[metric_id]

        def _noop_llm(messages):
            return '{"score": null, "reasoning": "No model configured."}'

        fn = llm_fn or _noop_llm

        results: List[CaseEvalResult] = []
        for c in cases:
            case_dict = {
                "question": c.question,
                "answer": c.answer,
                "expected_answer": c.expected_answer,
            }
            try:
                filled = _render_prompt(rec.prompt, case_dict)
                raw = fn([{"role": "user", "content": filled}])
                parsed = _parse_judge_response(raw)
                results.append(
                    CaseEvalResult(
                        question=c.question,
                        answer=c.answer,
                        expected_answer=c.expected_answer,
                        score=parsed.get("score"),
                        reasoning=parsed.get("reasoning", ""),
                    )
                )
            except Exception as exc:
                results.append(
                    CaseEvalResult(
                        question=c.question,
                        answer=c.answer,
                        expected_answer=c.expected_answer,
                        error=str(exc),
                    )
                )

        scored = [r.score for r in results if r.score is not None]
        avg = round(sum(scored) / len(scored), 4) if scored else None

        return EvaluateMetricResponse(
            metric_id=metric_id,
            name=rec.name,
            results=results,
            avg_score=avg,
        )

    def to_summary(self, rec: _MetricRecord) -> MetricSummary:
        return MetricSummary(
            metric_id=rec.metric_id,
            name=rec.name,
            description=rec.description,
            status=rec.status,
            created_at=rec.created_at,
        )

    def to_detail(self, rec: _MetricRecord) -> MetricDetail:
        return MetricDetail(
            metric_id=rec.metric_id,
            name=rec.name,
            description=rec.description,
            status=rec.status,
            created_at=rec.created_at,
            prompt=rec.prompt,
        )
