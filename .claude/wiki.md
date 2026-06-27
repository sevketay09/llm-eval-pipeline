# PROJECT WIKI — llm-eval-pipeline

## Architecture
FastAPI backend + React frontend (Vite/TS) + Streamlit UI for LLM evaluation with HITL review.
- Core: Async evaluation orchestration (EvalService), WebSocket real-time progress, JSON report persistence.
- Tracing (G1 — DONE): Online trace ingest SDK (tracing/sdk.py), deterministic sampler (tracing/sampler.py), TraceStore (in-memory, asyncio lock, FIFO evict), 4 REST endpoints (/api/traces/*).
- Red-team (G15 — DONE): Standalone redteam/ module, 5 attack categories, 13 templates, heuristic scorer, injectable model_fn/score_fn.
- Custom metrics / RAG eval / Failure clustering (G3/G8/G10 — DONE): Custom metric judge prompts, RAG component metrics, KMeans clustering analysis.
- Frontend (React): Dashboard, RunEval, Results, DatasetStudio, HitlReview, Models, Traces (G11 — DONE), RedTeam (G15 — DONE), Playground (G14 — DONE), CustomMetrics/RagEval/FailureClustering (G3/G8/G10 — DONE).
- Data flow: Test suite → model adapter → evaluator → metric aggregation → report → HITL annotation feedback.
- Python 3.9 compat: `from __future__ import annotations` on api/services/* and api/routers/websocket.py. scipy/sklearn mocked in conftest.py.

## Key Files
- `api/main.py` — FastAPI create_app(); all routers included here.
- `api/routers/traces.py` — POST /ingest, GET /, GET /{id}, POST /{id}/eval (G1).
- `api/routers/custom_metrics.py`, `api/services/custom_metric_service.py`, `api/schemas/custom_metrics.py` — G3 API.
- `api/routers/rag_eval.py`, `api/services/rag_eval_service.py`, `api/schemas/rag_eval.py` — G8 API.
- `api/routers/failure_clustering.py`, `api/services/failure_clustering_service.py`, `api/schemas/failure_clustering.py` — G10 API.
- `api/routers/redteam.py`, `api/services/redteam_service.py`, `api/schemas/redteam.py` — G15 API.
- `api/routers/evaluations.py`, `websocket.py`, `hitl.py`, `models.py`, `results.py`, `custom_datasets.py`, `experiments.py`.
- `api/services/trace_service.py` — TraceStore: asyncio lock, FIFO evict at 10k, tag/run_id filter.
- `api/services/eval_service.py` — Run orchestration, progress subscribers.
- `tracing/sdk.py` — EvalTrace, Span, EvalTracer (contextvar stacking), @trace decorator, ConsoleExporter, HttpExporter.
- `tracing/sampler.py` — OnlineSampler: MD5 hash-based deterministic sampling.
- `redteam/store.py`, `redteam/generator.py`, `redteam/scorer.py`, `redteam/runner.py` — G15: standalone module.
- `analysis/conv_simulator.py` — G9: Conversation simulation, trajectory eval.
- `analysis/rag_eval.py` — RAG metrics: context precision/recall, faithfulness, answer relevance (G8).
- `analysis/failure_clustering.py` — KMeans clustering with lazy sklearn imports (G10).
- `evaluators/custom_metric.py` — NL→judge prompt for custom metrics (G3).
- `reports/share.py` — G13: Shareable HTML report, permalink, embed.
- `web/src/pages/Traces.tsx`, `Playground.tsx`, `RedTeam.tsx`, `CustomMetrics.tsx`, `RagEval.tsx`, `FailureClustering.tsx` — React UI pages.
- `web/src/components/*` — Shared UI layer (PageHeader, Card, Button, Badge, ScoreBar, EmptyState, Field/Input/Textarea/Select, Spinner, Toast/useToast, CommandPalette, HelpHint) on top of index.css design system. All pages use this; no per-page hardcoded colors.
- `web/src/nav.ts` — Shared nav data (navGroups/navItems) consumed by App rail + CommandPalette.
- `web/src/api/client.ts` — All API clients.
- `conftest.py` — scipy/sklearn/numpy mock for contract test suite.

## Decisions
- Standalone module pattern: tracing/, redteam/, analysis/ have zero imports from api/; api/ imports one-way only.
- Lazy sklearn imports in analysis/failure_clustering.py to avoid numpy 2.x binary compat issues; conftest.py mocks sklearn with pure-numpy KMeans fallback.
- Custom metrics use NL→judge prompt chain (no explicit model dependency); injectable llm_fn for testing.
- RAG eval follows analysis/ pattern: standalone scoring functions (context_precision, context_recall, faithfulness, answer_relevance).
- TraceStore is in-memory (no DB) — max 10k traces, FIFO evict. Sufficient for online eval MVP.
- @trace decorator uses contextvars (not threading.local) — async-safe span stacking.
- Contract test suite is scipy-free by design; scipy/sklearn mocked in conftest.py.
- .gitignore: docs/, .claude/, .env, logs/, reports/, annotations/ all excluded from repo.

## Test Baseline
343 passed (290 baseline + 53 G3/G8/G10 contracts). Command:
```
pytest test_ci_gate_contracts.py test_synthetic_dataset_contracts.py \
  test_run_diff_contracts.py test_arena_elo_contracts.py \
  test_rag_eval_contracts.py test_conv_simulator_contracts.py \
  test_share_report_contracts.py test_tracing_sdk_contracts.py \
  test_sampler_contracts.py test_trace_service_contracts.py \
  test_traces_router_contracts.py test_redteam_contracts.py \
  test_custom_metrics_router_contracts.py test_rag_eval_router_contracts.py \
  test_failure_clustering_router_contracts.py -q
```

## Unfinished / Next
- G2, G4–G7 — Low priority features, backlog.
- Trace → Eval feedback loop: TraceStore ingests but not yet surfaced in HITL or reports.
- Frontend redesign (logs/frontend-redesign-plan-27-06-2026.md): Faz 1 DONE (6 off-brand pages → design system + web/src/components/). Faz 2 DONE (App.tsx nav grouped into Evaluate/Analyze/Configure; mobile off-canvas drawer + topbar hamburger; tablet icon-strip with title tooltips). Faz 3 DONE (CommandPalette Cmd+K + ⌘K rail trigger via nav.ts; ToastProvider/useToast wraps App in main.tsx, 5 tool pages converted setError→toast; empty-state CTAs on Dashboard/Results → /run; HelpHint "?" popover via PageHeader help prop on RAG/Failures/Red-Team/Custom Metrics). Faz 4 DONE (FailureClustering "From report" dropdown via resultsApi + "Paste JSON"/Load-example tab toggle; Skeleton component + Dashboard load skeletons + FailureClustering report-list skeleton; a11y — global :focus-visible ring, all expandable rows now role=button/tabIndex/aria-expanded + Enter/Space keyboard, icon-button aria-labels; responsive — table-shell overflow-x scroll). Frontend redesign plan fully complete.
