# PROJECT WIKI — llm-eval-pipeline

## Architecture
FastAPI backend + React frontend (Vite/TS) + Streamlit UI for LLM evaluation with HITL review.
- Core: Async evaluation orchestration (EvalService), WebSocket real-time progress, JSON report persistence.
- Tracing (G1 — DONE): Online trace ingest SDK (tracing/sdk.py), deterministic sampler (tracing/sampler.py), TraceStore (in-memory, asyncio lock, FIFO evict), 4 REST endpoints (/api/traces/*).
- Red-team (G15 — DONE): Standalone redteam/ module, 5 attack categories, 13 templates, heuristic scorer, injectable model_fn/score_fn.
- Frontend (React): Dashboard, RunEval, Results, DatasetStudio, HitlReview, Models, Traces (G11 — DONE), RedTeam (G15 — DONE).
- Data flow: Test suite → model adapter → evaluator → metric aggregation → report → HITL annotation feedback.
- Python 3.9 compat: `from __future__ import annotations` on api/services/* and api/routers/websocket.py. scipy mocked in conftest.py.

## Key Files
- `api/main.py` — FastAPI create_app(); all routers included here.
- `api/routers/traces.py` — POST /ingest, GET /, GET /{id}, POST /{id}/eval.
- `api/routers/redteam.py`, `api/services/redteam_service.py`, `api/schemas/redteam.py` — G15 API (POST/GET /api/redteam, /run, /results).
- `api/routers/evaluations.py`, `websocket.py`, `hitl.py`, `models.py`, `results.py`, `custom_datasets.py`.
- `api/services/trace_service.py` — TraceStore: asyncio lock, FIFO evict at 10k, tag/run_id filter.
- `api/services/eval_service.py` — Run orchestration, progress subscribers.
- `tracing/sdk.py` — EvalTrace, Span, EvalTracer (contextvar stacking), @trace decorator, ConsoleExporter, HttpExporter.
- `tracing/sampler.py` — OnlineSampler: MD5 hash-based deterministic sampling.
- `redteam/store.py`, `redteam/generator.py`, `redteam/scorer.py`, `redteam/runner.py` — G15: standalone module, 5 attack categories, 13 templates, heuristic scorer.
- `analysis/conv_simulator.py` — G9: Conversation simulation, trajectory eval (42 tests).
- `reports/share.py` — G13: Shareable HTML report, permalink, embed (35 tests).
- `web/src/pages/Traces.tsx` — G11: Live trace terminal UI (span tree, badges, eval button).
- `web/src/pages/RedTeam.tsx` — G15: Red-team React UI.
- `web/src/api/client.ts` — All API clients: modelsApi, evaluationsApi, resultsApi, tracesApi, hitlApi, customDatasetsApi, redteamApi.
- `conftest.py` — scipy/numpy mock for contract test suite.

## Decisions
- Standalone module pattern: tracing/ and redteam/ have zero imports from api/; api/ imports one-way only.
- redteam/ follows experiments/ pattern: injectable model_fn/score_fn, in-memory store, FIFO evict at 200.
- TraceStore is in-memory (no DB) — max 10k traces, FIFO evict. Sufficient for online eval MVP.
- @trace decorator uses contextvars (not threading.local) — async-safe span stacking.
- HttpExporter._send() is injectable — testable without real HTTP.
- Contract test suite is scipy-free by design; scipy mocked in conftest.py.
- .gitignore: docs/, .claude/, .env, logs/, reports/, annotations/ all excluded from repo.

## Test Baseline
290 passed (241 baseline + 49 G15 contracts). Command:
```
pytest test_ci_gate_contracts.py test_synthetic_dataset_contracts.py \
  test_run_diff_contracts.py test_arena_elo_contracts.py \
  test_rag_eval_contracts.py test_conv_simulator_contracts.py \
  test_share_report_contracts.py test_tracing_sdk_contracts.py \
  test_sampler_contracts.py test_trace_service_contracts.py \
  test_traces_router_contracts.py test_redteam_contracts.py -q
```

## Unfinished / Next
- G14 — Prompt playground (High difficulty, next)
- Trace → Eval feedback loop: TraceStore ingests but not yet surfaced in HITL or reports.
- OnlineSampler not yet wired into EvalService lifecycle.
