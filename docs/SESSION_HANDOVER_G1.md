# SESSION HANDOVER — G1: Online Eval + Trace Ingest
# Tarih: 27-06-2026

---

## Bu session'da ne tamamlandı

| Madde | Dosya | Test |
|-------|-------|------|
| G9 — Conversation Simulator | `analysis/conv_simulator.py` | 42 |
| G13 — Shareable Report + Embed | `reports/share.py` | 35 |
| **Toplam (scipy-free suite)** | | **172** |

Test komutu:
```bash
pytest test_ci_gate_contracts.py test_synthetic_dataset_contracts.py \
  test_run_diff_contracts.py test_arena_elo_contracts.py \
  test_rag_eval_contracts.py test_conv_simulator_contracts.py \
  test_share_report_contracts.py -q
# → 172 passed
```

---

## Hedef: G1 — Online Eval + Trace Ingest

### Ne yapılacak (kısa özet)

Şu an pipeline yalnız **batch** çalışıyor — kullanıcı bir kez koşturup gidiyor.
G1, prod uygulamasını **@eval.trace decorator** ile instrument edip
canlı trace'leri pipeline'a OTLP-uyumlu endpoint'e göndermeyi sağlar.
Langfuse/Braintrust/Phoenix'in retention gücü bu eksende.

---

## Mevcut API mimarisi (okumadan önce bil)

```
api/
  main.py           ← FastAPI create_app(); router'ları burada include_router ile ekle
  routers/
    evaluations.py  ← /api/evaluations/...
    results.py      ← /api/results/...
    websocket.py    ← WS eval progress
    hitl.py
    models.py
    custom_datasets.py
    __init__.py     ← router'ları export eder
  schemas/
    evaluations.py  ← EvalRunRequest, EvalRunStatus, ReportSummary (statistical_comparison var)
  services/
    eval_service.py
    report_service.py
```

`TraceSpanResult` ve `EvalCase` zaten **`utils/result_models.py`** içinde mevcut.
OTLP / OpenTelemetry bağımlılığı YOK — sıfırdan ekleyeceksin.

---

## G1 için önerilen mimari (3 katman)

### Katman 1 — SDK (standalone, sıfır bağımlılık)
**`tracing/sdk.py`**

```python
@eval.trace(name="my_rag", tags=["prod"])
def my_rag_fn(query: str) -> str:
    ...
```

- `EvalTrace` dataclass: `trace_id`, `name`, `tags`, `spans: List[Span]`, `start_ts`, `end_ts`
- `Span` dataclass: `span_id`, `parent_span_id`, `type` (LLM/TOOL/RETRIEVER/AGENT), `input`, `output`, `latency_ms`, `metadata`
- Decorator `@eval.trace` → contextvar-tabanlı span stacking (thread-safe)
- `EvalTracer` sınıfı: `start_span()` / `end_span()` / `flush(exporter)` context manager
- `ConsoleExporter` (stdout JSON) — LLM gerekmez, test edilebilir
- `HttpExporter(endpoint)` — prod; gerçek HTTP yapmaz, sadece arayüz

### Katman 2 — Ingest endpoint (API'ya eklenti)
**`api/routers/traces.py`** + **`api/services/trace_service.py`**

```
POST /api/traces/ingest          ← tek trace veya batch (list)
GET  /api/traces                 ← liste; ?run_id=... ?tag=...
GET  /api/traces/{trace_id}      ← tek trace detayı
POST /api/traces/{trace_id}/eval ← bu trace'i async eval et → EvalRunStatus
```

- `TraceStore` — bellek içi dict (`trace_id → EvalTrace`), opsiyonel JSON dump
- Schema: `TraceIngestRequest`, `TraceListResponse`, `TraceDetail` → `api/schemas/traces.py`
- **Önemli:** `api/routers/__init__.py`'a `traces_router` ekle + `api/main.py`'da `include_router`

### Katman 3 — Online sampler
**`tracing/sampler.py`**

- `OnlineSampler(rate=0.1)` → her 10 trace'den birini eval'e gönderir
- `sample(trace) -> bool` — deterministik (hash tabanlı), test edilebilir

---

## Kritik kural: Report contract bozulmasın

Tüm modüller bu yapıya uyar — yeni trace endpoint'i **dokunmaz**:
```json
{
  "models": {
    "<model_key>": {
      "overall_metrics": {"weighted_score": ..., "latency_p95": ...},
      "tests": {"<test_name>": {"summary": {"overall_score": ...}}}
    }
  },
  "summary": {"model_comparison": {...}},
  "statistical_comparison": {...}
}
```

Trace'ler **ayrı store**'da tutulur. Mevcut `/api/results/reports/` endpoint'ine dokunma.

---

## Standalone module pattern (bozma!)

`tracing/sdk.py` ve `tracing/sampler.py`:
- `api/`, `utils/`, `adapters/` import yok
- Injectable: `exporter` callable
- Argparse CLI
- Offline contract testleri: fake exporter + fake HTTP

`api/routers/traces.py` ve `api/services/trace_service.py`:
- Sadece `api/` kendi içinden import edebilir
- `tracing/` → `api/` import edilebilir (tek yön)

---

## Uygulama sırası (slice by slice)

**Slice 1 — `tracing/sdk.py` (standalone, bağımsız)**
Önce bunu yap. API'ya dokunmaz. Test edilebilir.
- `Span`, `EvalTrace`, `EvalTracer`, `@trace` decorator, `ConsoleExporter`
- Contract test: `test_tracing_sdk_contracts.py` (~20 test)

**Slice 2 — `tracing/sampler.py`**
Kısa, standalone.
- `OnlineSampler(rate, seed)` + deterministik `sample(trace_id)` + `should_eval(trace)`
- Contract test: `test_sampler_contracts.py` (~8 test)

**Slice 3 — `api/schemas/traces.py` + `api/services/trace_service.py`**
Store + business logic. API'ya henüz bağlanmaz.
- `TraceStore` (in-memory + disk dump), CRUD, tag filter
- Contract test: `test_trace_service_contracts.py` (~12 test)

**Slice 4 — `api/routers/traces.py` (FastAPI router)**
En son. Önceki slice'lar tamam olmadan başlama.
- 4 endpoint (ingest/list/detail/eval)
- `api/routers/__init__.py` + `api/main.py` güncelle
- Contract test: router-level TestClient testleri (~10 test)

**Her slice sonrası:** `pytest <new_test> + 172 eski test` → hepsi geçmeli.

---

## Dikkat noktaları

1. **`contextvar` (threading.local değil)** — span stacking için. Async-safe.
2. **`trace_id` = `uuid4().hex`** — dışarıdan da set edilebilir (OTLP trace_id ile uyum).
3. **`HttpExporter`** gerçek HTTP yapmamalı contract testlerde → inject edilebilir `_send(payload)` metodu.
4. **`POST /api/traces/ingest`** liste de kabul etmeli (batch ingest). `TraceIngestRequest = Union[EvalTrace, List[EvalTrace]]`.
5. **`TraceStore`** thread-safe olmalı (asyncio lock). Boyut limiti: max 10_000 trace (FIFO evict).

---

## Dosyalar: önce oku, sonra yaz

Session başında şu sırayla oku:
1. `utils/result_models.py` (mevcut TraceSpanResult yapısı)
2. `api/main.py` + `api/routers/__init__.py` (router ekleme pattern'i)
3. `api/schemas/evaluations.py` (schema convention)
4. `api/services/eval_service.py` (service pattern)
5. `docs/growth-opportunities.md` (G1 orijinal tanım)
6. `logs/development-27-06-2026.md` (bu session'da yapılanlar)

Codegraph sorgusu için: `codegraph explore "TraceSpanResult EvalCase result_models"` ve `codegraph explore "FastAPI router include evaluations api main"`

---

## Kalan G'ler (G1 sonrası)

| Madde | Zorluk | Not |
|-------|--------|-----|
| G11 — Live trace terminal UI | Orta | Frontend; G1 tamamlanınca frontend trace store'u tüketir |
| G14 — Prompt playground | Yüksek | Ayrı session |
| G15 — Auto red-team | Yüksek | Ayrı session |

---

## Test baseline

```bash
pytest test_ci_gate_contracts.py test_synthetic_dataset_contracts.py \
  test_run_diff_contracts.py test_arena_elo_contracts.py \
  test_rag_eval_contracts.py test_conv_simulator_contracts.py \
  test_share_report_contracts.py -q
# HEDEF: 172 passed, 0 failed
```

Her G slice'ı sonrası bu komutu çalıştır. Kırmızı görürsen dur, önce düzelt.

---

## Çalışma kuralları (CLAUDE.md'den)

- Token disiplini: broad scan yok. CodeGraph first.
- Standalone pattern: her yeni modül bağımsız test edilebilir.
- Daily log: `logs/development-27-06-2026.md` — her onaylı değişikten sonra 4 cümle.
- Caveman mode: kısa, dense.
- Suite kırmadan devam et.
