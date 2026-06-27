# PROJECT WIKI — llm-eval-pipeline

## Architecture
FastAPI backend + Streamlit UI for LLM evaluation with human-in-the-loop (HITL) review.
- Core: Evaluation orchestration (eval_service), async API with WebSocket real-time progress, report persistence.
- Tracing (G1): Online trace ingestion (api/routers/traces.py) feeding evaluation metrics with SDK (EvalTracer, Span) and sampling strategy (OnlineSampler).
- UI: Streamlit pages (human review, HITL analytics) connected to annotation storage; API routers (evaluations, websocket, hitl, models, traces).
- Data flow: Test suite definition → model adapter → evaluator → metric aggregation → report generation → HITL annotation feedback.

## Key Files
- `/main.py` — CLI entry point; loads suites from config/tests.yaml.
- `/api/routers/evaluations.py`, `/traces.py`, `/websocket.py` — FastAPI endpoints (run evals, stream progress, ingest traces).
- `/api/services/eval_service.py` — Main orchestration; manages runs, subscribers.
- `/api/services/trace_service.py`, `/tracing/sdk.py`, `/tracing/sampler.py` — Online trace capture and sampling.
- `/api/services/report_service.py` — Report CRUD, markdown/HTML rendering, metric aggregation.
- `/analysis/conv_simulator.py` (G9) — Conversation simulation, trajectory evaluation, suite runner.
- `/reports/share.py` (G13) — Report sharing/export.
- `/pages/4_Human_Review.py` — Streamlit HITL interface; annotation manager, filtering, export.
- `/config/tests.yaml` — Test suite definitions.

## Decisions
- Async API + WebSocket for live eval progress streaming.
- Service layer (dependency injection via FastAPI Depends) decouples routers from logic.
- JSON report storage with multi-run versioning (report → runs array).
- Singleton EvalService shared across routers; TraceStore for async trace persistence.
- AnnotationManager for HITL feedback storage; agreement metrics (LLM vs human).
- Trace schema (TraceSchema, TraceDetail) enables span-level telemetry during eval.

## Unfinished
- G1: Trace ingest router created; trace_service skeleton exists. Needs: backend storage (SQLite/PG), indexing, query DSL, eval-trace linking in EvalService.
- Trace → Eval feedback loop: traces ingested but not yet queried/surfaced in HITL UI or reports.
- Sampler policy: OnlineSampler defined; needs integration into eval lifecycle (when/how to sample).
