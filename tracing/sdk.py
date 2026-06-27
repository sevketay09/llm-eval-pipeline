"""
tracing/sdk.py — Standalone trace decorator + span collection.
No imports from api/, utils/, adapters/.
"""
from __future__ import annotations

import functools
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Span:
    span_id: str
    name: str
    type: str  # LLM | TOOL | RETRIEVER | AGENT | GENERIC
    input: Any
    output: Any
    latency_ms: float
    parent_span_id: Optional[str] = None
    start_ts: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "type": self.type,
            "input": self.input,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "start_ts": self.start_ts,
            "metadata": self.metadata,
        }


@dataclass
class EvalTrace:
    trace_id: str
    name: str
    tags: List[str] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "tags": list(self.tags),
            "spans": [s.to_dict() for s in self.spans],
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "metadata": self.metadata,
        }


# ── Context vars (async-safe, thread-safe) ───────────────────────────────────

_active_trace: ContextVar[Optional[EvalTrace]] = ContextVar("_active_trace", default=None)
_span_stack: ContextVar[List[str]] = ContextVar("_span_stack", default=[])


# ── Exporters ─────────────────────────────────────────────────────────────────

class ConsoleExporter:
    def export(self, trace: EvalTrace) -> None:
        print(json.dumps(trace.to_dict(), indent=2, default=str))


class HttpExporter:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def _send(self, payload: Dict[str, Any]) -> None:
        import urllib.request
        data = json.dumps(payload, default=str).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)

    def export(self, trace: EvalTrace) -> None:
        self._send(trace.to_dict())


# ── SpanHandle (manual start/end API) ────────────────────────────────────────

@dataclass
class SpanHandle:
    span_id: str
    parent_span_id: Optional[str]
    name: str
    type: str
    input: Any
    metadata: Dict[str, Any]
    _trace: Optional[EvalTrace]
    _start: float

    def end(self, output: Any = None) -> Span:
        latency = (time.time() - self._start) * 1000
        s = Span(
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            type=self.type,
            input=self.input,
            output=output,
            latency_ms=latency,
            start_ts=self._start,
            metadata=self.metadata,
        )
        if self._trace is not None:
            self._trace.spans.append(s)
        stack = list(_span_stack.get())
        if self.span_id in stack:
            stack.remove(self.span_id)
        _span_stack.set(stack)
        return s


# ── EvalTracer ────────────────────────────────────────────────────────────────

class EvalTracer:
    def __init__(
        self,
        name: str,
        tags: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ):
        self.name = name
        self.tags = tags or []
        self.trace_id = trace_id or uuid.uuid4().hex
        self._trace: Optional[EvalTrace] = None

    def __enter__(self) -> "EvalTracer":
        self._trace = EvalTrace(
            trace_id=self.trace_id,
            name=self.name,
            tags=list(self.tags),
            start_ts=time.time(),
        )
        _active_trace.set(self._trace)
        _span_stack.set([])
        return self

    def __exit__(self, *_) -> None:
        if self._trace is not None:
            self._trace.end_ts = time.time()
        _active_trace.set(None)
        _span_stack.set([])

    def flush(self, exporter) -> None:
        if self._trace is not None:
            exporter.export(self._trace)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        type: str = "GENERIC",
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[List[Any], None, None]:
        """Context manager span. Capture output via: `with tracer.span(...) as out: out[0] = result`"""
        current_trace = _active_trace.get()
        stack = list(_span_stack.get())
        parent_id = stack[-1] if stack else None
        span_id = uuid.uuid4().hex
        start = time.time()
        stack.append(span_id)
        _span_stack.set(stack)
        output_holder: List[Any] = [None]
        try:
            yield output_holder
        finally:
            latency = (time.time() - start) * 1000
            s = Span(
                span_id=span_id,
                parent_span_id=parent_id,
                name=name,
                type=type,
                input=input,
                output=output_holder[0],
                latency_ms=latency,
                start_ts=start,
                metadata=metadata or {},
            )
            if current_trace is not None:
                current_trace.spans.append(s)
            new_stack = list(_span_stack.get())
            if span_id in new_stack:
                new_stack.remove(span_id)
            _span_stack.set(new_stack)

    def start_span(
        self,
        name: str,
        *,
        type: str = "GENERIC",
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanHandle:
        current_trace = _active_trace.get()
        stack = list(_span_stack.get())
        parent_id = stack[-1] if stack else None
        span_id = uuid.uuid4().hex
        stack.append(span_id)
        _span_stack.set(stack)
        return SpanHandle(
            span_id=span_id,
            parent_span_id=parent_id,
            name=name,
            type=type,
            input=input,
            metadata=metadata or {},
            _trace=current_trace,
            _start=time.time(),
        )


# ── @trace decorator ──────────────────────────────────────────────────────────

def trace(
    name: Optional[str] = None,
    *,
    tags: Optional[List[str]] = None,
    exporter=None,
) -> Callable:
    """
    @trace(name="my_rag", tags=["prod"])
    def my_rag_fn(query: str) -> str: ...
    """
    def decorator(fn: Callable) -> Callable:
        _name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = EvalTracer(name=_name, tags=tags or [])
            with tracer:
                result = fn(*args, **kwargs)
            if exporter is not None:
                tracer.flush(exporter)
            return result

        wrapper._tracer_name = _name  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="tracing/sdk demo")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        exp = ConsoleExporter()

        @trace(name="demo_rag", tags=["demo"], exporter=exp)
        def demo_rag(query: str) -> str:
            return f"answer to: {query}"

        demo_rag("hello world")
