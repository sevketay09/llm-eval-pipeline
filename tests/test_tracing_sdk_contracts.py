"""Contract tests for tracing/sdk.py — no LLM, no network."""
from __future__ import annotations

import time
import pytest

from tracing.sdk import (
    ConsoleExporter,
    EvalTrace,
    EvalTracer,
    HttpExporter,
    Span,
    SpanHandle,
    _active_trace,
    _span_stack,
    trace,
)


# ── Span ──────────────────────────────────────────────────────────────────────

class TestSpan:
    def test_span_creation_fields(self):
        s = Span(span_id="abc", name="s1", type="LLM", input="q", output="a", latency_ms=10.0)
        assert s.span_id == "abc"
        assert s.name == "s1"
        assert s.type == "LLM"
        assert s.latency_ms == 10.0

    def test_span_parent_default_none(self):
        s = Span(span_id="x", name="s", type="TOOL", input=None, output=None, latency_ms=0.0)
        assert s.parent_span_id is None

    def test_span_to_dict_keys(self):
        s = Span(span_id="abc", name="s1", type="RETRIEVER", input="q", output="a", latency_ms=5.0)
        d = s.to_dict()
        for k in ("span_id", "parent_span_id", "name", "type", "input", "output", "latency_ms", "start_ts", "metadata"):
            assert k in d, f"missing key: {k}"

    def test_span_to_dict_values(self):
        s = Span(span_id="abc", name="s1", type="AGENT", input="q", output="a", latency_ms=3.0, parent_span_id="p1")
        d = s.to_dict()
        assert d["span_id"] == "abc"
        assert d["parent_span_id"] == "p1"
        assert d["output"] == "a"

    def test_span_metadata_default_empty(self):
        s = Span(span_id="x", name="s", type="GENERIC", input=None, output=None, latency_ms=0.0)
        assert s.metadata == {}


# ── EvalTrace ─────────────────────────────────────────────────────────────────

class TestEvalTrace:
    def test_trace_defaults(self):
        t = EvalTrace(trace_id="t1", name="fn")
        assert t.tags == []
        assert t.spans == []
        assert t.end_ts is None

    def test_trace_to_dict_structure(self):
        t = EvalTrace(trace_id="t1", name="fn", tags=["prod"])
        d = t.to_dict()
        assert d["trace_id"] == "t1"
        assert d["name"] == "fn"
        assert d["tags"] == ["prod"]
        assert isinstance(d["spans"], list)

    def test_trace_to_dict_with_span(self):
        t = EvalTrace(trace_id="t1", name="fn")
        s = Span(span_id="s1", name="step", type="GENERIC", input=None, output=None, latency_ms=1.0)
        t.spans.append(s)
        d = t.to_dict()
        assert len(d["spans"]) == 1
        assert d["spans"][0]["span_id"] == "s1"


# ── EvalTracer context manager ────────────────────────────────────────────────

class TestEvalTracer:
    def test_context_sets_active_trace(self):
        tracer = EvalTracer(name="t")
        with tracer:
            assert _active_trace.get() is not None
        assert _active_trace.get() is None

    def test_records_end_ts(self):
        tracer = EvalTracer(name="t")
        with tracer:
            pass
        assert tracer._trace.end_ts is not None
        assert tracer._trace.end_ts >= tracer._trace.start_ts

    def test_custom_trace_id(self):
        tracer = EvalTracer(name="t", trace_id="myid123")
        with tracer:
            pass
        assert tracer._trace.trace_id == "myid123"

    def test_tags_propagated(self):
        tracer = EvalTracer(name="t", tags=["a", "b"])
        with tracer:
            pass
        assert tracer._trace.tags == ["a", "b"]

    def test_flush_calls_exporter(self):
        exported = []

        class FakeExp:
            def export(self, t): exported.append(t)

        tracer = EvalTracer(name="t")
        with tracer:
            pass
        tracer.flush(FakeExp())
        assert len(exported) == 1
        assert exported[0].name == "t"

    def test_span_ctx_adds_span(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("step1", type="LLM", input="q") as out:
                out[0] = "answer"
        assert len(tracer._trace.spans) == 1
        s = tracer._trace.spans[0]
        assert s.name == "step1"
        assert s.output == "answer"
        assert s.type == "LLM"

    def test_span_latency_positive(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("step") as out:
                time.sleep(0.01)
                out[0] = "done"
        assert tracer._trace.spans[0].latency_ms > 5

    def test_nested_spans_parent_id(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("outer", type="AGENT") as _:
                with tracer.span("inner", type="LLM") as _:
                    pass
        spans = tracer._trace.spans
        assert len(spans) == 2
        inner = next(s for s in spans if s.name == "inner")
        outer = next(s for s in spans if s.name == "outer")
        assert inner.parent_span_id == outer.span_id

    def test_nested_spans_stack_cleared_after(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("s1") as _:
                pass
        assert _span_stack.get() == []

    def test_start_span_end_adds_span(self):
        tracer = EvalTracer(name="t")
        with tracer:
            h = tracer.start_span("manual", type="RETRIEVER", input="doc")
            h.end(output="result")
        assert len(tracer._trace.spans) == 1
        s = tracer._trace.spans[0]
        assert s.name == "manual"
        assert s.output == "result"
        assert s.type == "RETRIEVER"

    def test_multiple_spans(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("a") as _: pass
            with tracer.span("b") as _: pass
            with tracer.span("c") as _: pass
        assert len(tracer._trace.spans) == 3

    def test_span_metadata(self):
        tracer = EvalTracer(name="t")
        with tracer:
            with tracer.span("s", metadata={"model": "gpt-4"}) as _:
                pass
        assert tracer._trace.spans[0].metadata["model"] == "gpt-4"


# ── Exporters ─────────────────────────────────────────────────────────────────

class TestConsoleExporter:
    def test_outputs_json(self, capsys):
        t = EvalTrace(trace_id="cx", name="fn_console")
        ConsoleExporter().export(t)
        out = capsys.readouterr().out
        assert "cx" in out
        assert "fn_console" in out

    def test_output_contains_trace_id_key(self, capsys):
        t = EvalTrace(trace_id="cx2", name="fn2")
        ConsoleExporter().export(t)
        out = capsys.readouterr().out
        assert "trace_id" in out


class TestHttpExporter:
    def test_injectable_send(self):
        sent = []
        exp = HttpExporter(endpoint="http://fake/ingest")
        exp._send = lambda payload: sent.append(payload)
        t = EvalTrace(trace_id="hx", name="fn")
        exp.export(t)
        assert len(sent) == 1
        assert sent[0]["trace_id"] == "hx"

    def test_endpoint_stored(self):
        exp = HttpExporter(endpoint="http://localhost:4317/v1/traces")
        assert exp.endpoint == "http://localhost:4317/v1/traces"

    def test_export_sends_name(self):
        sent = []
        exp = HttpExporter(endpoint="http://fake")
        exp._send = lambda p: sent.append(p)
        t = EvalTrace(trace_id="hx", name="my_fn")
        exp.export(t)
        assert sent[0]["name"] == "my_fn"


# ── @trace decorator ──────────────────────────────────────────────────────────

class TestTraceDecorator:
    def test_returns_original_result(self):
        @trace(name="fn")
        def fn(x): return x * 2
        assert fn(5) == 10

    def test_calls_exporter(self):
        exported = []

        class FakeExp:
            def export(self, t): exported.append(t)

        @trace(name="fn", exporter=FakeExp())
        def fn(): return "ok"

        fn()
        assert len(exported) == 1

    def test_uses_fn_name_when_no_name(self):
        exported = []

        class FakeExp:
            def export(self, t): exported.append(t)

        @trace(exporter=FakeExp())
        def my_function(): return "ok"

        my_function()
        assert exported[0].name == "my_function"

    def test_preserves_fn_name(self):
        @trace(name="named")
        def my_fn(): pass
        assert my_fn.__name__ == "my_fn"

    def test_tags_propagated(self):
        exported = []

        class FakeExp:
            def export(self, t): exported.append(t)

        @trace(name="fn", tags=["prod", "v2"], exporter=FakeExp())
        def fn(): return None

        fn()
        assert exported[0].tags == ["prod", "v2"]

    def test_no_exporter_does_not_crash(self):
        @trace(name="fn")
        def fn(): return 42
        assert fn() == 42

    def test_exporter_receives_eval_trace(self):
        exported = []

        class FakeExp:
            def export(self, t): exported.append(t)

        @trace(name="fn", exporter=FakeExp())
        def fn(): return "x"

        fn()
        assert isinstance(exported[0], EvalTrace)
