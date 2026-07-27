from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def _copy_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _pick_first_string(payload: Mapping[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


@dataclass(frozen=True)
class TraceSpanResult:
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    span_type: str = "system"
    name: Optional[str] = None
    status: str = "completed"
    duration_ms: float = 0.0
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    metric_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TraceSpanResult":
        known_keys = {
            "span_id",
            "parent_span_id",
            "span_type",
            "type",
            "name",
            "status",
            "duration_ms",
            "duration",
            "input_summary",
            "output_summary",
            "metric_results",
            "error",
            "metadata",
        }
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        metric_results = payload.get("metric_results", [])
        return cls(
            span_id=_pick_first_string(payload, ["span_id"]),
            parent_span_id=_pick_first_string(payload, ["parent_span_id"]),
            span_type=_pick_first_string(payload, ["span_type", "type"]) or "system",
            name=_pick_first_string(payload, ["name"]),
            status=_pick_first_string(payload, ["status"]) or "completed",
            duration_ms=_coerce_float(payload.get("duration_ms", payload.get("duration", 0.0))),
            input_summary=_pick_first_string(payload, ["input_summary"]),
            output_summary=_pick_first_string(payload, ["output_summary"]),
            metric_results=list(metric_results) if isinstance(metric_results, list) else [],
            error=_pick_first_string(payload, ["error"]),
            metadata=_copy_mapping(payload.get("metadata")),
            extra=extra,
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "span_type": self.span_type,
            "status": self.status,
        }
        if self.span_id is not None:
            payload["span_id"] = self.span_id
        if self.parent_span_id is not None:
            payload["parent_span_id"] = self.parent_span_id
        if self.name is not None:
            payload["name"] = self.name
        if self.duration_ms:
            payload["duration_ms"] = self.duration_ms
        if self.input_summary is not None:
            payload["input_summary"] = self.input_summary
        if self.output_summary is not None:
            payload["output_summary"] = self.output_summary
        if self.metric_results:
            payload["metric_results"] = list(self.metric_results)
        if self.error is not None:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class TraceResult:
    trace_id: Optional[str] = None
    spans: List[TraceSpanResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TraceResult":
        raw_spans = payload.get("spans", [])
        spans: List[TraceSpanResult] = []
        if isinstance(raw_spans, list):
            for item in raw_spans:
                if isinstance(item, Mapping):
                    spans.append(TraceSpanResult.from_payload(item))
        return cls(
            trace_id=_pick_first_string(payload, ["trace_id"]),
            spans=spans,
            summary=_copy_mapping(payload.get("summary")),
            metadata=_copy_mapping(payload.get("metadata")),
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "summary": dict(self.summary),
        }
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        if self.spans:
            payload["spans"] = [span.to_payload() for span in self.spans]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    input_text: Optional[str] = None
    model_answer: Optional[str] = None
    scores: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    structured_output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    trace: Optional[TraceResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CaseResult":
        trace_payload = payload.get("trace")
        known_keys = {
            "id",
            "test_id",
            "case_id",
            "question",
            "prompt",
            "input_text",
            "input",
            "model_answer",
            "response",
            "output",
            "answer",
            "scores",
            "details",
            "latency",
            "structured_output",
            "error",
            "trace",
            "metadata",
        }
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        return cls(
            case_id=_pick_first_string(payload, ["case_id", "test_id", "id"]) or "unknown",
            input_text=_pick_first_string(payload, ["question", "prompt", "input_text", "input"]),
            model_answer=_pick_first_string(payload, ["model_answer", "response", "output", "answer"]),
            scores=_copy_mapping(payload.get("scores")),
            details=_copy_mapping(payload.get("details")),
            latency=_coerce_float(payload.get("latency", 0.0)),
            structured_output=_copy_mapping(payload.get("structured_output")),
            error=_pick_first_string(payload, ["error"]),
            trace=TraceResult.from_payload(trace_payload) if isinstance(trace_payload, Mapping) else None,
            metadata=_copy_mapping(payload.get("metadata")),
            extra=extra,
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "case_id": self.case_id,
        }
        if self.input_text is not None:
            payload["input_text"] = self.input_text
        if self.model_answer is not None:
            payload["model_answer"] = self.model_answer
        if self.scores:
            payload["scores"] = dict(self.scores)
        if self.details:
            payload["details"] = dict(self.details)
        if self.latency:
            payload["latency"] = self.latency
        if self.structured_output:
            payload["structured_output"] = dict(self.structured_output)
        if self.error is not None:
            payload["error"] = self.error
        if self.trace is not None:
            payload["trace"] = self.trace.to_payload()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class TestResult:
    test_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], fallback_name: str) -> "TestResult":
        known_keys = {"test_name", "results", "summary", "metadata"}
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        raw_results = payload.get("results", [])
        serialized_results: List[Dict[str, Any]] = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, Mapping):
                    serialized_results.append(CaseResult.from_payload(item).to_payload())
                else:
                    serialized_results.append({"value": item})
        return cls(
            test_name=_pick_first_string(payload, ["test_name"]) or fallback_name,
            results=serialized_results,
            summary=_copy_mapping(payload.get("summary")),
            metadata=_copy_mapping(payload.get("metadata")),
            extra=extra,
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "test_name": self.test_name,
            "results": list(self.results),
            "summary": dict(self.summary),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class ModelRunResult:
    model_key: str
    model_name: str
    provider: str
    runtime_parameters: Dict[str, Any] = field(default_factory=dict)
    tests: Dict[str, Any] = field(default_factory=dict)
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], fallback_key: str) -> "ModelRunResult":
        tests_payload = payload.get("tests", {})
        serialized_tests: Dict[str, Any] = {}
        if isinstance(tests_payload, Mapping):
            for test_name, test_payload in tests_payload.items():
                if isinstance(test_payload, Mapping):
                    serialized_tests[test_name] = TestResult.from_payload(test_payload, test_name).to_payload()
                else:
                    serialized_tests[test_name] = test_payload

        known_keys = {
            "model_key",
            "model_name",
            "provider",
            "runtime_parameters",
            "tests",
            "overall_metrics",
        }
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        return cls(
            model_key=_pick_first_string(payload, ["model_key"]) or fallback_key,
            model_name=_pick_first_string(payload, ["model_name"]) or fallback_key,
            provider=_pick_first_string(payload, ["provider"]) or "unknown",
            runtime_parameters=_copy_mapping(payload.get("runtime_parameters")),
            tests=serialized_tests,
            overall_metrics=_copy_mapping(payload.get("overall_metrics")),
            extra=extra,
        )

    @classmethod
    def empty(
        cls,
        *,
        model_key: str,
        model_name: str,
        provider: str,
        runtime_parameters: Optional[Dict[str, Any]] = None,
    ) -> "ModelRunResult":
        return cls(
            model_key=model_key,
            model_name=model_name,
            provider=provider,
            runtime_parameters=dict(runtime_parameters or {}),
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "provider": self.provider,
            "runtime_parameters": dict(self.runtime_parameters),
            "tests": dict(self.tests),
            "overall_metrics": dict(self.overall_metrics),
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class RunResult:
    timestamp: str
    models: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    trends: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunResult":
        models_payload = payload.get("models", {})
        serialized_models: Dict[str, Any] = {}
        if isinstance(models_payload, Mapping):
            for model_key, model_payload in models_payload.items():
                if isinstance(model_payload, Mapping):
                    serialized_models[model_key] = ModelRunResult.from_payload(model_payload, model_key).to_payload()
                else:
                    serialized_models[model_key] = model_payload

        known_keys = {
            "timestamp",
            "models",
            "summary",
            "trends",
            "comparisons",
            "run_metadata",
        }
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        return cls(
            timestamp=_pick_first_string(payload, ["timestamp"]) or "",
            models=serialized_models,
            summary=_copy_mapping(payload.get("summary")),
            trends=_copy_mapping(payload.get("trends")),
            comparisons=_copy_mapping(payload.get("comparisons")),
            run_metadata=_copy_mapping(payload.get("run_metadata")),
            extra=extra,
        )

    @classmethod
    def empty(
        cls,
        *,
        timestamp: str,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunResult":
        return cls(
            timestamp=timestamp,
            run_metadata=dict(run_metadata or {}),
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "models": dict(self.models),
            "summary": dict(self.summary),
            "trends": dict(self.trends),
            "comparisons": dict(self.comparisons),
            "run_metadata": dict(self.run_metadata),
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


def serialize_run_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return RunResult.from_payload(payload).to_payload()


def serialize_test_result_payload(test_name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return TestResult.from_payload(payload, test_name).to_payload()