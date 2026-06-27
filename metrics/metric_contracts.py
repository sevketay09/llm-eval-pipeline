from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_unit_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    provider: str
    group: Optional[str] = None
    normalized_value: Optional[float] = None
    success: Optional[bool] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "provider": self.provider,
        }
        if self.group is not None:
            payload["group"] = self.group
        if self.normalized_value is not None:
            payload["normalized_value"] = self.normalized_value
        if self.success is not None:
            payload["success"] = self.success
        if self.reason:
            payload["reason"] = self.reason
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.raw_payload:
            payload["raw_payload"] = self.raw_payload
        return payload


def build_metric_result(
    name: str,
    value: Any,
    *,
    provider: str,
    group: Optional[str] = None,
    normalized_value: Optional[float] = None,
    success: Optional[bool] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raw_payload: Optional[Dict[str, Any]] = None,
) -> Optional[MetricResult]:
    if not _is_number(value):
        return None

    metric_value = float(value)
    normalized = normalized_value
    if normalized is None:
        normalized = _normalize_unit_score(metric_value)

    return MetricResult(
        name=name,
        value=metric_value,
        provider=provider,
        group=group,
        normalized_value=normalized,
        success=success,
        reason=reason,
        metadata=metadata or {},
        raw_payload=raw_payload,
    )


def build_metric_results_from_mapping(
    scores: Mapping[str, Any],
    *,
    provider: str,
    group: Optional[str] = None,
) -> List[MetricResult]:
    metric_results: List[MetricResult] = []
    for name, value in scores.items():
        metric_result = build_metric_result(
            name,
            value,
            provider=provider,
            group=group,
        )
        if metric_result is not None:
            metric_results.append(metric_result)
    return metric_results


def build_metric_result_from_payload(
    name: str,
    payload: Optional[Mapping[str, Any]],
    *,
    provider: str,
    group: Optional[str] = None,
    score_key: str = "score",
    reason_key: Optional[str] = "reasoning",
    metadata_keys: Optional[Mapping[str, str]] = None,
    raw_payload_key: Optional[str] = None,
    raw_payload_fallback: bool = True,
) -> Optional[MetricResult]:
    if not isinstance(payload, Mapping):
        return None

    metadata: Dict[str, Any] = {}
    if metadata_keys:
        for metadata_name, payload_key in metadata_keys.items():
            metadata_value = payload.get(payload_key)
            if metadata_value is not None:
                metadata[metadata_name] = metadata_value

    raw_payload: Optional[Dict[str, Any]] = None
    if raw_payload_key is not None:
        raw_candidate = payload.get(raw_payload_key)
        if isinstance(raw_candidate, Mapping):
            raw_payload = dict(raw_candidate)
    elif raw_payload_fallback:
        raw_payload = dict(payload)

    reason = payload.get(reason_key) if reason_key else None
    return build_metric_result(
        name,
        payload.get(score_key),
        provider=provider,
        group=group,
        reason=reason,
        metadata=metadata,
        raw_payload=raw_payload,
    )


def build_metric_results_from_payload_mapping(
    payloads: Mapping[str, Any],
    *,
    provider: str,
    group: Optional[str] = None,
    metric_names: Optional[Iterable[str]] = None,
    score_key: str = "score",
    reason_key: Optional[str] = "reasoning",
    metadata_keys: Optional[Mapping[str, str]] = None,
    raw_payload_key: Optional[str] = None,
    raw_payload_fallback: bool = True,
) -> List[MetricResult]:
    if not isinstance(payloads, Mapping):
        return []

    metric_results: List[MetricResult] = []
    names = metric_names if metric_names is not None else payloads.keys()
    for metric_name in names:
        metric_result = build_metric_result_from_payload(
            metric_name,
            payloads.get(metric_name),
            provider=provider,
            group=group,
            score_key=score_key,
            reason_key=reason_key,
            metadata_keys=metadata_keys,
            raw_payload_key=raw_payload_key,
            raw_payload_fallback=raw_payload_fallback,
        )
        if metric_result is not None:
            metric_results.append(metric_result)
    return metric_results


def serialize_metric_results(
    metric_results: Iterable[Optional[MetricResult]],
) -> List[Dict[str, Any]]:
    return [metric_result.to_dict() for metric_result in metric_results if metric_result is not None]