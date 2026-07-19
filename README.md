# LLM Evaluation Pipeline

<div align="center">

**Production-grade LLM evaluation, observability & red-teaming platform**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/tests-364%20passed-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Compare models · Watch live traces · Experiment with prompts · Attack your own guardrails*

[Quick Start](#quick-start) · [Architecture](#architecture) · [React UI](#react-ui) · [API](#rest-api) · [Eval Datasets](#eval-datasets)

**[🇹🇷 Türkçe README](README.tr.md)**

</div>

---

## Overview

LLM Evaluation Pipeline is a comprehensive evaluation framework that combines batch model comparison, live trace ingestion, a prompt playground, human-in-the-loop review and automated red-teaming in a single platform.

**What is it for?**

- Systematically comparing alternatives before picking a production model
- Instrumenting your own RAG/agent application and watching live traces
- Running prompt versions side by side on the same dataset (A/B)
- Automatically probing model resistance to jailbreaks and prompt injection
- Building and calibrating domain-specific custom metrics
- Producing a failure taxonomy that auto-clusters low-quality outputs
- Blocking quality regressions in CI/CD when models are updated

## Screenshots

| Dashboard | Prompt Playground | Auto Red-Team |
|-----------|-------------------|---------------|
| ![Dashboard](assets/dashboard.png) | ![Playground](assets/playground.png) | ![Red-Team](assets/redteam.png) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     React + Vite Frontend                        │
│  Dashboard · Run · Results · Traces · Playground · Red-Team      │
│  HITL · Datasets · Custom Metrics · RAG Eval · Failures          │
│                   (port 5173 dev / web/dist prod)                │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP + WebSocket
┌─────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend  (port 8001)                  │
│  /api/evaluations  /api/traces  /api/experiments  /api/redteam  │
│  /api/custom-metrics  /api/rag-eval  /api/failure-clustering     │
│  /api/hitl  /api/results  /api/models  /ws/progress             │
│              api/routers/ + api/services/                        │
└──────┬──────────────┬────────────────┬──────────────────────────┘
       │              │                │
┌──────▼──────┐ ┌─────▼───────┐ ┌─────▼──────────────────────────┐
│  pipeline_  │ │  tracing/   │ │     Standalone modules         │
│  runner.py  │ │  sdk.py     │ │  experiments/  redteam/        │
│  evaluators/│ │  sampler.py │ │  analysis/     datagen/        │
│  adapters/  │ │  TraceStore │ │  evaluators/custom_metric.py   │
└─────────────┘ └─────────────┘ └────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn + WebSocket |
| Frontend | React 18 + Vite + TypeScript |
| LLM clients | openai SDK (OpenAI/Azure/OpenRouter/vLLM/Ollama), anthropic |
| Observability | Custom tracing SDK (EvalTracer, @trace decorator, OTLP-like) |
| LLM-as-judge | Provider-agnostic judge evaluators (quality/agent/groundedness) |
| Data processing | pandas, numpy, scikit-learn (lazy import) |
| Data models | Pydantic v2 |
| Configuration | YAML + python-dotenv |
| Containers | Docker + Compose |

---

## Quick Start

### Development

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd web && npm install && cd ..

# Backend + frontend together
make dev
```

| Address | Service |
|---------|---------|
| `http://localhost:5173` | React frontend (dev) |
| `http://localhost:8001` | FastAPI backend |
| `http://localhost:8001/docs` | Swagger UI |
| `ws://localhost:8001/ws/progress/<run_id>` | Real-time progress |

### Production Build

```bash
cd web && npm run build          # creates web/dist/
set -a && source .env && set +a
uvicorn api.main:app --host 0.0.0.0 --port 8001
# → http://localhost:8001 (API + frontend)
```

### Docker

```bash
cp .env.example .env             # fill in your keys
docker compose up --build        # http://localhost:8001
```

### 60-Second Demo (no API keys needed)

```bash
make demo            # offline smoke eval with the mock model → populated dashboard
make demo-docker     # same demo inside the Docker image
```

The demo run writes a real report into `reports/`; open the dashboard afterwards with `make dev` and explore the results.

### Command Line

```bash
python main.py --models gpt-4o --suite smoke                        # quick smoke
python main.py --models gpt-4o qwen-3-30b --suite full              # full comparison
python main.py --models qwen-3-30b --suite mcp_only                 # agentic only
```

### Makefile Targets

```bash
make dev              # backend + frontend
make dev-backend      # FastAPI only
make dev-frontend     # Vite only
make build-frontend   # production build
make check-api        # health check
make start-debug      # Docker debug stack
```

---

## Modules

The project is built from standalone modules joined by one-way dependencies. Each module owns its dataclasses and in-memory store; the `api/` layer exposes them as REST endpoints.

### tracing/ — Online Eval & Trace Ingestion

Instrument your own LLM application and stream live traces into the platform.

```python
from tracing.sdk import EvalTracer, HttpExporter

tracer = EvalTracer(exporters=[HttpExporter("http://localhost:8001/api/traces/ingest")])

@tracer.trace("rag-query")
def answer(question: str) -> str:
    with tracer.span("retriever"):
        docs = retrieve(question)
    with tracer.span("llm"):
        return generate(docs, question)
```

- `EvalTracer` + `@trace` decorator: contextvar-based span stacking (async-safe)
- `OnlineSampler`: MD5 hash-based deterministic sampling
- `TraceStore`: asyncio lock, FIFO eviction (10k), tag/run_id filters
- Ingestion: `POST /api/traces/ingest` → `GET /api/traces` → `POST /api/traces/{id}/eval`

### experiments/ — Prompt Playground

Run multiple prompt versions on the same dataset and diff them case by case.

```python
from experiments.store import PromptVariant, ExperimentCase, make_experiment
from experiments.runner import ExperimentRunner

runner = ExperimentRunner(model_fn=my_llm)
exp = make_experiment("v1 vs v2", variants=[...], dataset=[...])
results = runner.run(exp)
```

- `ExperimentRunner`: injectable `model_fn` / `score_fn`
- `compute_diff`: improved / regressed / stable / missing verdicts
- `ExperimentStore`: 500 records, FIFO eviction
- API: `POST /api/experiments` → `/run` → `/compare`

### redteam/ — Auto Red-Team

Stress a system prompt with 13 templates across 5 attack categories.

```python
from redteam.generator import generate_attacks
from redteam.runner import RedTeamRunner

attacks = generate_attacks("You are a helpful assistant.", ["jailbreak", "prompt_injection"])
runner = RedTeamRunner(model_fn=my_llm)
results = runner.run_session(session)
```

| Category | Description |
|----------|-------------|
| `prompt_injection` | "Ignore previous instructions…" variations |
| `jailbreak` | DAN, developer override, base model appeal |
| `persona_override` | Role switching, "evil twin" |
| `boundary_test` | PII requests, harmful content |
| `role_confusion` | Admin override, developer command |

- Heuristic scorer: compliance marker vs refusal marker detection
- `passed=True` → the model resisted the attack
- API: `POST /api/redteam` → `/run` → `/results`

### evaluators/custom_metric.py — Custom Metrics

Automatic judge prompt generation from a natural-language description:

```python
from evaluators.custom_metric import generate_judge_prompt, evaluate_with_custom_metric

prompt = generate_judge_prompt("Rate how empathetic the response is, 0-1")
result = evaluate_with_custom_metric(case, prompt, llm_fn=my_llm)
# → {"score": 0.85, "reasoning": "..."}
```

- Template-based prompt without an LLM (noop fallback)
- `calibrate_metric`: Pearson correlation against a human label set
- API: `POST /api/custom-metrics` → `/{id}/evaluate`

### analysis/rag_eval.py — RAG Component Evaluation

Separate retriever failures from generator failures:

```python
from analysis.rag_eval import evaluate_rag_case

result = evaluate_rag_case({
    "question": "...", "contexts": [...], "answer": "..."
})
# → context_precision, context_recall, faithfulness, answer_relevance, fault_component
```

| Metric | Measures |
|--------|----------|
| `context_precision` | Share of relevant chunks in context |
| `context_recall` | How much of the answer the context covers |
| `faithfulness` | Answer's grounding in the context |
| `answer_relevance` | How well the answer addresses the question |
| `fault_component` | `retriever` / `generator` / `both` / `none` |

- API: `POST /api/rag-eval`

### analysis/failure_clustering.py — Failure Taxonomy

Cluster low-scoring cases and label them automatically:

```python
from analysis.failure_clustering import compute_failure_summary

summary = compute_failure_summary(report, threshold=0.6)
# → {"total_failures": 42, "clusters": [...], "model_breakdown": {...}}
```

- KMeans clustering (injectable `embed_fn`, lazy sklearn import)
- Keyword-based automatic cluster labels
- API: `POST /api/failure-clustering`

### analysis/conv_simulator.py — Conversation Simulator

Run N-turn scenarios between a persona-driven synthetic user and your agent:

```python
from analysis.conv_simulator import run_simulation_suite

results = run_simulation_suite(
    agent_fn=my_agent,
    personas=[neutral, demanding, confused],
    turns=5
)
```

- `goal_completion`, `coherence`, `efficiency` metrics
- CLI: `python -m analysis.conv_simulator --demo`

### analysis/significance.py — Statistical Significance

Tell whether a score difference is noise or a real effect:

```python
from analysis.significance import compute_significance
results = compute_significance("report.json", alpha=0.05, seed=42)
# → bootstrap CI, paired t-test, Wilcoxon, Cohen's d_z
```

- CLI: `python -m analysis.significance REPORT.json --format markdown`

### reports/share.py — Shareable Reports

Dark-mode HTML report, social card meta tags, embeddable leaderboard:

```python
from reports.share import build_share_report
html = build_share_report(report, title="Q2 Model Comparison")
```

- Gzip+base64 permalink: reopen with `decode_permalink(url_hash)`
- CLI: `python -m reports.share REPORT.json --format html`

### datagen/ — Synthetic Dataset Generation

Generate golden Q/A datasets from documents; solve the cold-start problem:

```bash
python -m datagen.generate \
    --source docs/guide.md \
    --project "E-commerce bot" \
    --model gpt-4o \
    --output eval_datasets/generated/my_dataset.json
```

- `chunk_text` → LLM prompt → Q/A pair normalization → nondeterministic case filter
- Turkish/English source support

---

## React UI

`web/` — React 18 + Vite + TypeScript SPA. The production build in `web/dist/` is served by FastAPI.

| Page | Route | What it does |
|------|-------|--------------|
| **Dashboard** | `/` | Metric overview, latest run comparison, trends |
| **Run Evaluation** | `/run` | Pick models + suite, live progress over WebSocket |
| **Results** | `/results` | Report browser, model scores, AI commentary |
| **Live Traces** | `/traces` | Live trace list, span tree, eval button |
| **Prompt Playground** | `/playground` | Prompt A/B, dataset editor, diff table |
| **Auto Red-Team** | `/redteam` | Attack a system prompt with 13 templates |
| **Custom Metrics** | `/custom-metrics` | NL description → judge prompt → case evaluation |
| **RAG Eval** | `/rag-eval` | Question + context + answer → component scores |
| **Failure Clustering** | `/failures` | Paste report JSON → cluster taxonomy |
| **HITL Review** | `/review` | Review queue, annotation, trace queue |
| **Dataset Studio** | `/datasets` | Dataset upload, synthetic generation |
| **Models** | `/models` | Add/edit/delete models |

---

## REST API

Full Swagger UI: `http://localhost:8001/docs`

### Evaluations

| Method | Path | |
|--------|------|-|
| `POST` | `/api/evaluations/run` | Start a new eval |
| `POST` | `/api/evaluations/runs/{id}/cancel` | Cancel |
| `GET` | `/api/evaluations/suites` | List suites |
| `WS` | `/ws/progress/{run_id}` | Real-time progress |

### Traces & Observability

| Method | Path | |
|--------|------|-|
| `POST` | `/api/traces/ingest` | Ingest a trace |
| `GET` | `/api/traces` | List traces (tag/run_id filters) |
| `GET` | `/api/traces/{id}` | Trace detail + span tree |
| `POST` | `/api/traces/{id}/eval` | Queue a trace for eval |

### Prompt Experiments

| Method | Path | |
|--------|------|-|
| `POST` | `/api/experiments` | Create an experiment |
| `GET` | `/api/experiments` | List |
| `GET` | `/api/experiments/{id}` | Detail |
| `POST` | `/api/experiments/{id}/run` | Run (202) |
| `GET` | `/api/experiments/{id}/compare` | Variant diff |

### Auto Red-Team

| Method | Path | |
|--------|------|-|
| `POST` | `/api/redteam` | Create a session |
| `GET` | `/api/redteam` | List |
| `GET` | `/api/redteam/{id}` | Detail |
| `POST` | `/api/redteam/{id}/run` | Run attacks (202) |
| `GET` | `/api/redteam/{id}/results` | Results |

### Custom Metrics

| Method | Path | |
|--------|------|-|
| `POST` | `/api/custom-metrics` | Create a metric (prompt auto-generated) |
| `GET` | `/api/custom-metrics` | List |
| `GET` | `/api/custom-metrics/{id}` | Detail + prompt |
| `POST` | `/api/custom-metrics/{id}/evaluate` | Evaluate cases |

### RAG & Failure Analysis

| Method | Path | |
|--------|------|-|
| `POST` | `/api/rag-eval` | RAG component scores |
| `POST` | `/api/failure-clustering` | Report → cluster taxonomy |

### HITL & Results

| Method | Path | |
|--------|------|-|
| `GET` | `/api/hitl/pending` | Items awaiting review |
| `POST` | `/api/hitl/review` | Save an annotation |
| `GET` | `/api/hitl/calibration` | Judge calibration metrics |
| `GET` | `/api/results/reports` | List reports |
| `GET` | `/api/results/reports/{filename}` | Report detail |
| `POST` | `/api/custom-datasets` | Upload a dataset |
| `POST` | `/api/custom-datasets/generate` | Generate a synthetic dataset |

---

## CI Integration (Eval-as-CI)

Turns evaluation output into a quality gate.

### CLI

```bash
python -m ci.gate reports/ci_report.json \
    --config config/ci_gate.yaml \
    --baseline reports/baseline.json \
    --format markdown
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Gate passed |
| `1` | Threshold violation |
| `2` | I/O error |

`--format badge` → produces shields.io endpoint JSON.

### config/ci_gate.yaml

```yaml
weighted_score_min: 0.75
max_latency_p95: 10.0
fail_on_test_error: true
tests:
  turkish_grammar:
    min_score: 0.80
  function_calling:
    min_score: 0.75
regression:
  max_weighted_drop: 0.05
  max_test_drop: 0.10
```

### pytest Integration

```python
from ci.pytest_plugin import load_report, assert_gate, assert_no_regression, assert_test_score

def test_quality():
    assert_gate(load_report("reports/ci_report.json"))

def test_no_regression():
    assert_no_regression(
        load_report("reports/ci_report.json"),
        load_report("reports/baseline.json")
    )
```

### GitHub Actions

`.github/workflows/llm-eval.yml` ships with the repo. Composite action:

```yaml
- name: LLM Eval Gate
  uses: ./.github/actions/llm-eval-gate
  with:
    report: reports/ci_report.json
    config: config/ci_gate.yaml
    baseline: reports/baseline.json      # optional
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

The action writes the gate summary to `$GITHUB_STEP_SUMMARY`, posts a PR comment, produces `llm-eval-badge.json` and fails the job on violations.

---

## Scoring

### Layer 1 — Per Item

**LLM-as-judge (categorical):**

| Label | Score |
|-------|-------|
| `TAM_DOGRU` (fully correct) | 1.0 |
| `KISMEN_DOGRU` (partially correct) | 0.5 |
| `YANLIS` (wrong) | 0.0 |

No numeric 1-10 scale — the categorical design avoids verbosity bias.

**Agent judge (agentic tests):** `task_adherence`, `tool_call_accuracy`, `response_completeness`, `intent_resolution` — provider-agnostic LLM-as-judge (`evaluators/agent_judge.py`). Judge parse failures return `None` and are excluded from aggregates instead of polluting them as fake zeros.

**Quality judge:** `coherence`, `fluency`, `relevance`, `groundedness` (1-5 → normalized 0-1, `evaluators/quality_judge.py`).

### Layer 2 — Per Test overall_score

| Test type | Formula |
|-----------|---------|
| `qa`, `turkish_*`, `fintech_*` | `(FULL×1.0 + PARTIAL×0.5) / total` |
| `mcp_tool_use`, `agentic_workflows` | `agentic_pack_aggregate` |
| `function_calling` | `avg(overall_lenient)` |
| `function_calling_chain` | `0.7×tool_coverage + 0.3×order_score` |
| `rag_test`, `needle_in_haystack` | `avg_rag_quality` |
| `adversarial_security` | `safety_rate` |
| `embedding_sts` | `spearman_correlation` |
| `embedding_retrieval` | `NDCG@10` |
| `multi_turn`, `stress_tests` | `avg_context_retention` |

### Layer 3 — Per Model weighted_score

```
weighted_score = Σ(test_overall_score × weight) / Σ(weight)
```

Weights are defined in `config/tests.yaml`. `error_rate`, `latency` and `tokens_per_second` are infrastructure metrics and never enter `weighted_score`.

---

## Tests

All contract tests live in `tests/`, discovered via `pytest.ini`.

```bash
pytest                                          # whole suite
pytest tests/test_redteam_router_contracts.py  # single file
pytest -k "experiments"                         # filtered
```

**Current baseline: 364 contract tests passing.**

Root `conftest.py` for test isolation:
- scipy (numpy 2.x binary compat) → `MagicMock`
- sklearn (numpy 2.x binary compat) → pure-numpy KMeans mock

---

## Supported Models

| Provider | Examples |
|----------|----------|
| OpenAI | GPT-4o, GPT-5 |
| Azure OpenAI | GPT-4o (PTU/PR), GPT-4.1 |
| Anthropic | Claude Sonnet 4.x |
| OpenRouter | Any hosted model behind one API key |
| vLLM (on-premise) | Qwen-3-30B, Mistral-Small-3.1, LLaMA-3-70B |
| Ollama (local) | llama3, mistral, gemma2, phi3 |
| LM Studio (local) | Any GGUF model |

```yaml
# config/models.yaml example
models:
  gpt-4o:
    provider: openai
    api_key: ${AZURE_OPENAI_KEY}
    base_url: ${AZURE_OPENAI_ENDPOINT}
    model_name: ${AZURE_OPENAI_DEPLOYMENT_NAME_PTU}
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true

  qwen-3-30b:
    provider: openai        # vLLM OpenAI-compatible endpoint
    base_url: ${VLLM_BASE_URL}
    api_key: dummy
    model_name: default
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true
```

---

## Environment Variables

Copy from `.env.example`:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME_PTU=gpt-4o

# OpenAI Direct
OPENAI_API_KEY=your-key

# Anthropic
ANTHROPIC_API_KEY=your-key

# OpenRouter
OPENROUTER_API_KEY=your-key

# On-premise vLLM
VLLM_BASE_URL=http://your-vllm-server:8000/v1
MISTRAL_VLLM_BASE_URL=http://your-mistral-server:8000/v1

# Local
OLLAMA_BASE_URL=http://localhost:11434/v1
LMSTUDIO_MODEL1_BASE_URL=http://localhost:1234/v1
```

---

## Eval Datasets

JSON test sets in 9 categories under `eval_datasets/`:

| Folder | Contents |
|--------|----------|
| `benchmark/` | Turkish grammar/reasoning/creativity/paraphrasing, PII, self-consistency, negative constraints |
| `agentic/` | Multi-step task planning, tool selection |
| `edge_cases/` | Adversarial (jailbreak/injection), edge case scenarios |
| `embedding/` | STS, cross-lingual STS, retrieval, hard-negative retrieval, domain clustering |
| `fintech/` | Fintech domain knowledge, financial calculations |
| `function_calling/` | Basic tool selection, parallel tools, tool chains, error recovery |
| `multi_turn/` | Context retention, long-context stress |
| `rag/` | RAG quality, needle-in-haystack |
| `regression/` | Golden set, recent-issue regression |
| `security/` | PII leakage, auth bypass, stress |

---

## Project Structure

```
llm-eval-pipeline/
├── api/
│   ├── main.py                   # App factory, CORS, lifespan
│   ├── routers/                  # evaluations, traces, experiments,
│   │                             # redteam, custom_metrics, rag_eval,
│   │                             # failure_clustering, hitl, models,
│   │                             # results, custom_datasets, websocket
│   ├── services/                 # EvalService, TraceService, RedTeamService,
│   │                             # ExperimentService, CustomMetricService,
│   │                             # RagEvalService, FailureClusteringService…
│   └── schemas/                  # Pydantic request/response models
├── tracing/
│   ├── sdk.py                    # EvalTracer, Span, @trace, exporters
│   └── sampler.py                # OnlineSampler (MD5 deterministic)
├── experiments/
│   ├── store.py                  # PromptVariant, ExperimentCase, ExperimentStore
│   ├── runner.py                 # ExperimentRunner (injectable model_fn)
│   └── differ.py                 # compute_diff → improved/regressed/stable
├── redteam/
│   ├── store.py                  # Attack, AttackResult, RedTeamSession
│   ├── generator.py              # 13 AttackTemplates, generate_attacks()
│   ├── scorer.py                 # Heuristic scorer (compliance vs refusal)
│   └── runner.py                 # RedTeamRunner (injectable model_fn)
├── analysis/
│   ├── rag_eval.py               # context_precision/recall/faithfulness/relevance
│   ├── failure_clustering.py     # KMeans clustering + keyword labels
│   ├── conv_simulator.py         # Persona-based synthetic user simulation
│   ├── significance.py           # Bootstrap CI, paired t-test, Cohen's d_z
│   ├── arena_elo.py              # Bradley-Terry/Elo pairwise leaderboard
│   └── run_diff.py               # Diff between two runs
├── evaluators/                   # 25 independent evaluators
│   ├── llm_judge.py              # Categorical LLM-as-judge
│   ├── quality_judge.py          # coherence/fluency/relevance/groundedness
│   ├── agent_judge.py            # task adherence, tool accuracy, completeness, intent
│   ├── groundedness_judge.py     # RAG faithfulness judge
│   ├── judge_utils.py            # Shared JSON parsing + retry for judges
│   ├── geval.py                  # G-Eval (fluency/coherence/relevance)
│   ├── custom_metric.py          # NL → judge prompt, calibrate, evaluate
│   └── ...                       # (hallucination, safety, adversarial, RAG…)
├── datagen/
│   └── generate.py               # Document → chunk → Q/A golden dataset
├── reports/
│   └── share.py                  # Shareable HTML report, permalink, embed
├── ci/
│   ├── gate.py                   # Quality gate (thresholds + regression)
│   └── pytest_plugin.py          # assert_gate, assert_weighted_score, …
├── web/
│   ├── src/
│   │   ├── pages/                # 12 React pages
│   │   └── api/client.ts         # All API client functions
│   └── dist/                     # Production build
├── tests/                        # 40 contract test files
│   └── test_*.py
├── adapters/
│   ├── unified_adapter.py        # Single LLM interface (all providers)
│   └── embedding_adapter.py
├── eval_datasets/                # Test datasets (JSON)
├── config/
│   ├── models.yaml
│   ├── tests.yaml                # Suite definitions + weights
│   └── ci_gate.yaml
├── pytest.ini
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

---

## Troubleshooting

**vLLM connection error:**
```bash
python -m vllm.entrypoints.openai.api_server --model MODEL_NAME --port 8000
```

**API key error:**
```bash
source .env && echo $OPENAI_API_KEY
```

**Frontend unreachable:** on a port clash Vite moves to the next port automatically; check the terminal output.

**Judge timeouts:** add `timeout: 60` to `config/models.yaml`.

**sklearn binary compat error:** `conftest.py` automatically mocks the numpy 2.x incompatibility; always run tests through `pytest`.

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

[⬆ Back to top](#llm-evaluation-pipeline)

</div>
