# LLM Evaluation Pipeline

<div align="center">

**Enterprise-grade LLM evaluation framework**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Bulut, on-premise ve yerel LLM'leri sistematik biçimde test edip karşılaştırın*

[Hızlı Başlangıç](#hızlı-başlangıç) • [Skor Hesaplama](#skor-hesaplama-mantığı) • [Test Suitleri](#test-suitleri) • [Mimari](#mimari)

</div>

---

## Genel Bakış

**LLM Evaluation Pipeline**, üretim kalitesinde LLM değerlendirmesi için kapsamlı bir framework'tür. Doğruluk, akıl yürütme kalitesi, Türkçe dil yetkinliği, araç kullanımı (function calling / MCP), agentic iş akışları ve güvenlik gibi boyutlarda modelleri yan yana karşılaştırır.

### Ne İçin Kullanılır?

- Üretim modelini seçmeden önce alternatifleri karşılaştırmak
- Belirli bir domain'de (fintech, hukuk, dil) hangi modelin daha iyi çalıştığını ölçmek
- Model güncellemelerinde kalite regresyonunu otomatik tespit etmek
- Human-in-the-Loop anotasyon döngüsünü yönetmek
- Azure AI Evaluation SDK ile agentic test skorları almak

---

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│                   React + Vite Frontend                 │
│  Dashboard │ RunEvaluation │ Results │ HitlReview │ ... │
│                   (port 5173 / dist/)                   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP + WebSocket
┌────────────────────────▼────────────────────────────────┐
│              FastAPI Backend  (port 8001)               │
│  /api/evaluations  /api/results  /api/hitl  /ws/progress│
│                api/routers/ + api/services/             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               pipeline_runner.py                        │
│   EvaluationPipeline → test tipi başına run_*_test()   │
│   → evaluators/ → reports/*.json                       │
└──────┬──────────────┬──────────────────┬────────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼──────┐  ┌───────▼──────────────┐
│  adapters/  │ │ evaluators/│  │   eval_datasets/     │
│  unified_   │ │ llm_judge  │  │  benchmark/ fintech/ │
│  adapter.py │ │ azure_agent│  │  agentic/ function_  │
│  embedding_ │ │ geval      │  │  calling/ multi_turn/│
│  adapter.py │ │ nlp_metrics│  │  rag/ embedding/ ... │
└─────────────┘ └────────────┘  └──────────────────────┘
```

### Teknoloji Yığını

| Katman | Teknoloji | Versiyon |
|--------|-----------|---------|
| **Backend API** | FastAPI + Uvicorn | 0.115+ |
| **Frontend** | React + Vite + TypeScript | React 18 |
| **LLM İstemcileri** | openai SDK (Azure/OpenAI/vLLM/Ollama), anthropic | 1.12+ |
| **Azure AI Evaluation** | `azure-ai-evaluation` (agentic evaluators) | 1.16+ |
| **NLP Metrikleri** | rouge-score, nltk, scipy | - |
| **Veri İşleme** | pandas, numpy, scikit-learn | - |
| **Real-time İletişim** | WebSocket (FastAPI native) | - |
| **Veri Modelleri** | Pydantic v2 | 2.5+ |
| **Konfigürasyon** | YAML + python-dotenv | - |
| **Konteyner** | Docker + Compose | - |

---

## Hızlı Başlangıç

### Geliştirme Ortamı

```bash
# Python bağımlılıkları
pip install -r requirements.txt

# Frontend bağımlılıkları
cd web && npm install && cd ..

# Backend + Frontend birlikte
make dev
```

**Varsayılan adresler:**
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8001`
- API Docs (Swagger): `http://localhost:8001/docs`
- WebSocket ilerleme: `ws://localhost:8001/ws/progress/<run_id>`

### Üretim Build

```bash
cd web && npm run build
set -a && source .env && set +a
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Docker

```bash
cp .env.example .env
# .env içini endpoint/key değerlerinizle doldurun
docker compose up --build
# Dashboard: http://localhost:8001
```

**Notlar:**
- Repoda key/endpoint saklamayın; tüm hassas değerler `.env` üzerinden okunur.
- `docker-compose.yml` içinde `./config:/app/config` mount edildiği için arayüzden eklenen modeller kalıcı olur.
- Sonuçlar `./reports` altında host üzerinde kalır (container silinse de kaybolmaz).

### Komut Satırı

```bash
# Hızlı smoke testi
python main.py --models gpt-4o --suite smoke

# Birden fazla model, tam değerlendirme
python main.py --models gpt-4o mistral-small-3.1 qwen-3-30b --suite full

# Sadece MCP + agentic testleri
python main.py --models qwen-3-30b --suite mcp_only
```

### Makefile Hedefleri

```bash
make dev              # Backend + frontend birlikte (önerilen)
make dev-backend      # Sadece FastAPI backend
make dev-frontend     # Sadece Vite frontend
make build-frontend   # dist/ için production build
make check-api        # API health check
make start-debug      # Docker debug stack (down + build + up)
make tail-logs        # Docker log takibi
```

---

## Skor Hesaplama Mantığı

Sistem üç katmanlı bir hesaplama yapar.

### Katman 1 — Item Bazında Değerlendirme (case score)

#### LLM-as-Judge (kategorik)

Birincil evaluator GPT-4o'dur. Yapılandırılmış JSON çıktı üretir:

```json
{"label": "TAM_DOGRU", "reasoning": "Yanıt doğru ve eksiksiz."}
```

Etiket → sayısal değer dönüşümü:

| Etiket | Puan | Anlam |
|--------|------|-------|
| `TAM_DOGRU` | 1.0 | Yanıt tam doğru ve eksiksiz |
| `KISMEN_DOGRU` | 0.5 | Kısmen doğru veya eksik |
| `YANLIS` | 0.0 | Yanlış, alakasız veya hallücine edilmiş |

Numeric 1-10 skalası yoktur; verbosity bias'tan kaçınmak için kategorik tasarım tercih edilmiştir.

#### Azure AI Evaluation — Agentic Evaluators

`mcp_tool_use` ve `agentic_workflows` testlerinde 4 ek evaluator devreye girer (`evaluators/azure_agent.py`, `azure-ai-evaluation>=1.16.0`):

| Evaluator | Ne Ölçer | Eşik |
|-----------|----------|------|
| `TaskAdherenceEvaluator` | Modelin göreve bağlı kalıp kalmadığı | ≥ 0.70 |
| `ToolCallAccuracyEvaluator` | Araç çağrılarının doğruluk oranı | ≥ 0.75 |
| `ResponseCompletenessEvaluator` | Yanıtın sorguyu tam karşılayıp karşılamadığı | ≥ 0.70 |
| `IntentResolutionEvaluator` | Kullanıcı niyetinin çözülüp çözülmediği | ≥ 0.70 |

API formatı: `TaskAdherence`, `ResponseCompleteness` ve `IntentResolution` `conversation={"messages": [...]}` dict alır; `ToolCallAccuracy` `query=str, response=[msgs]` parametreleri alır.

Bu 4 evaluator'ın çıktıları `agentic_pack` metrik grubunu oluşturur:

| Metrik Adı | Kaynak |
|------------|--------|
| `plan_adherence` | LLM judge plan skorlaması |
| `task_completion` | `TaskAdherence.score` |
| `tool_correctness` | `ToolCallAccuracy.score` |
| `step_efficiency` | Azure agent `aggregate_score` |
| `response_completeness` | `ResponseCompleteness.score` |
| `intent_resolution` | `IntentResolution.score` |
| `agentic_pack_aggregate` | Yukarıdakilerin normalize ortalaması |

#### NLP Metrikleri

- **ROUGE-1/2/L**: N-gram örtüşmesi
- **BLEU**: Çeviri kalitesi skoru
- **BERTScore** (opsiyonel, GPU): Semantik benzerlik

#### G-Eval (LLM tabanlı)

İnsan değerlendirmesini taklit eden boyutsal skorlama: `fluency`, `coherence`, `relevance`, `correctness`

---

### Katman 2 — Test Bazında `overall_score`

| Test Tipi | Formül |
|-----------|--------|
| `qa`, `fintech_*`, `turkish_*` | `(TAM×1.0 + KISMİ×0.5) / toplam_item` |
| `mcp_tool_use`, `agentic_workflows` | `agentic_pack_aggregate` |
| `multi_turn`, `stress_tests` | `avg_context_retention` |
| `function_calling` | `avg(overall_lenient)` |
| `function_calling_chain` | `0.7 × tool_coverage + 0.3 × order_score` |
| `rag_test`, `needle_in_haystack` | `avg_rag_quality` |
| `adversarial_security`, `security_tests` | `safety_rate` |
| `pii_detection` | `accuracy` (binary) |
| `self_consistency` | `avg_consistency_score` |
| `negative_constraints` | `compliance_rate` |
| `prompt_compression` | `avg_quality_score` |
| `edge_cases`, `regression_*` | `avg_behavior_score × (1 - error_rate)` |
| `embedding_sts`, `embedding_sts_crosslingual` | `spearman_correlation` |
| `embedding_retrieval`, `embedding_retrieval_hardneg` | `NDCG@10` |
| `language_mix` | LLM judge ortalaması |

---

### Katman 3 — Model Bazında `weighted_score` (Ana Skor)

```
weighted_score = Σ(test_overall_score × weight) / Σ(weight)
```

Test ağırlıkları (`config/tests.yaml`):

| Test | Ağırlık | Test | Ağırlık |
|------|---------|------|---------|
| `embedding_sts` | 0.10 | `fintech_knowledge` | 0.08 |
| `embedding_retrieval` | 0.10 | `fintech_calculations` | 0.08 |
| `embedding_retrieval_hardneg` | 0.08 | `function_calling` | 0.08 |
| `embedding_sts_crosslingual` | 0.08 | `pii_detection` | 0.06 |
| `turkish_grammar` | 0.07 | `adversarial_security` | 0.06 |
| `turkish_creativity` | 0.07 | `embedding_clustering_regulatory` | 0.06 |
| `turkish_expression_errors` | 0.07 | `language_mix` | 0.05 |
| `turkish_paraphrasing` | 0.07 | `mcp_tool_use` | 0.05 |
| `turkish_reasoning` | 0.07 | `self_consistency` | 0.05 |

### Yan Metrikler (Altyapı — Kalite Puanı Değil)

`weighted_score`'a girmeyen, ancak raporda görünen altyapı metrikleri:

| Metrik | Açıklama |
|--------|----------|
| `error_rate` | API hata oranı — `%0 = API çalışıyor`, kaliteyle ilgisiz |
| `timeout_rate` | Zaman aşımı oranı |
| `latency_avg` / `latency_p95` | Ortalama ve P95 gecikme (saniye) |
| `tokens_per_second` | Throughput (token/sn) |
| `score_stability` | `1 - std(test_scores)`: tutarlılık |
| `quality_latency_efficiency` | `weighted_score / avg_latency` |
| `judge_agreement_rate` | Judge uyuşma oranı |
| `total_input_tokens` / `total_output_tokens` | Maliyet hesabı için token sayıları |

---

## Test Suitleri

`config/tests.yaml` içinde tanımlıdır.

| Süit Adı | İçerdiği Testler | Kullanım Amacı |
|----------|-----------------|---------------|
| `mcp_only` | `mcp_tool_use`, `agentic_workflows` | MCP + agentic odaklı hızlı karşılaştırma |
| `fintech_only` | `fintech_knowledge`, `fintech_calculations`, `function_calling`, `agentic_workflows`, `mcp_tool_use`, `pii_detection` | Fintech domain testi |
| `embedding_full` | `embedding_sts`, `embedding_sts_crosslingual`, `embedding_retrieval`, `embedding_retrieval_hardneg`, `embedding_clustering`, `embedding_clustering_regulatory` | Embedding model kalite değerlendirmesi |
| `embedding_turkish` | `embedding_sts`, `embedding_retrieval`, `embedding_clustering` | Türkçe embedding odaklı |
| `advanced` | `multi_turn`, `multi_turn_stress`, `rag_test`, `adversarial_security`, `edge_cases`, `security_tests`, `stress_tests`, `regression_golden`, `regression_recent` | İleri düzey güvenlik + bağlam testleri |
| `benchmarks` | `mmlu`, `hellaswag`, `truthfulqa`, `humaneval`, `gsm8k` | Akademik benchmark karşılaştırması |
| `smoke` | Temel testler | Hızlı CI/CD gate |
| `full` | Tüm ağırlıklı testler | Tam değerlendirme |

---

## Desteklenen Modeller ve Sağlayıcılar

| Sağlayıcı | Tip | Örnekler |
|-----------|-----|---------|
| **Azure OpenAI** | Bulut | GPT-4o (PTU/PR), GPT-4.1, GPT-5.2 |
| **OpenAI** | Bulut | GPT-4o, GPT-5.2 |
| **Anthropic** | Bulut | Claude Sonnet 4.5 |
| **vLLM** | On-Premise (OpenAI uyumlu) | Qwen-3-30B, Mistral-Small-3.1, LLaMA-3-70B |
| **Ollama** | Yerel | llama3, mistral, gemma2, phi3 |
| **LM Studio** | Yerel (OpenAI uyumlu) | GGUF formatındaki herhangi bir model |

`config/models.yaml` içindeki kayıtlı modeller UI'dan doğrudan seçilebilir.

---

## Evaluator'lar

`evaluators/` dizinindeki her dosya bağımsız bir değerlendirme bileşenidir.

| Dosya | Görev |
|-------|-------|
| `llm_judge.py` | GPT-4o tabanlı kategorik judge (TAM/KISMİ/YANLIŞ) |
| `azure_agent.py` | Azure AI Evaluation SDK — 4 agentic evaluator |
| `azure_quality.py` | Azure AI Evaluation SDK — kalite metrikleri |
| `geval.py` | LLM tabanlı boyutsal G-Eval skorlaması |
| `nlp_metrics.py` | ROUGE, BLEU, BERTScore |
| `hallucination_eval.py` | Hallucination tespit ve puanlama |
| `accuracy_eval.py` | Exact/fuzzy match, sayısal tolerans |
| `safety_eval.py` | PII sızıntısı, güvenlik uyumu |
| `advanced_eval.py` | Needle-in-Haystack, Tool Error Recovery, Paralel Tool |
| `consistency_eval.py` | Self-consistency (çoklu çalıştırma + majority vote) |
| `negative_constraints_eval.py` | Negatif kısıtlama uyumu |
| `prompt_compression_eval.py` | Sıkıştırılmış prompt kalite ölçümü |
| `adversarial_eval.py` | Jailbreak / prompt injection direnci |
| `language_mix_eval.py` | Türkçe–İngilizce karışık dil değerlendirmesi |
| `pii_eval.py` | PII tespiti (binary accuracy) |
| `embedding_eval.py` | Embedding kalitesi (STS, retrieval, clustering) |
| `benchmark_eval.py` | MMLU, HellaSwag, TruthfulQA, HumanEval, GSM8K |
| `comparative_eval.py` | Model karşılaştırma yardımcıları |
| `human_feedback_eval.py` | HITL anotasyon entegrasyonu |
| `faithfulness.py` | RAG context faithfulness |
| `dynamic_function_eval.py` | Dinamik function calling değerlendirmesi |
| `error_recovery_eval.py` | Tool hata kurtarma değerlendirmesi |
| `needle_haystack_eval.py` | Uzun bağlamda bilgi arama |

---

## React UI

`web/` dizininde React + Vite + TypeScript ile geliştirilmiş SPA. Production build `web/dist/` olarak FastAPI tarafından serve edilir.

### Sayfalar

| Sayfa | Rota | Açıklama |
|-------|------|----------|
| **Dashboard** | `/` | Genel metrik özeti, son run karşılaştırması, trend grafikleri |
| **Run Evaluation** | `/run` | Model seçimi, süit seçimi, gerçek zamanlı ilerleme (WebSocket) |
| **Results** | `/results` | Rapor tarayıcısı; model skoru, AI commentary, span trace |
| **HITL Review** | `/hitl` | Human-in-the-Loop anotasyon ekranı |
| **Models** | `/models` | Model ekleme/silme/konfigürasyon |
| **Dataset Studio** | `/datasets` | Custom dataset yükleme ve yönetimi |

### HITL (Human-in-the-Loop) İş Akışı

1. Pipeline, düşük güvenli veya çelişkili sonuçları `reports/evaluations_store.json` kuyruğuna atar
2. `HitlReview.tsx` bekleyen item'ları listeler; sidebar'da Review Action + Queue Status gösterilir
3. Reviewer `TAM DOĞRU / KISMİ DOĞRU / YANLIŞ` etiketler, açıklama ekler
4. Onaylanan etiketler `evaluations_store.json`'a yazılır, sonraki raporda görünür

### AI Commentary

Her rapor için GPT-4o, modelin güçlü/zayıf yönlerini otomatik analiz eder. Prompt şunu açıkça ayırt eder:
- **Kalite Metrikleri**: `weighted_score`, judge skorları (0–1 arası; yüksek = iyi)
- **Altyapı Metrikleri**: `error_rate`, `latency` (`error_rate %0 = API çalışıyor`, kaliteyle ilgisiz)

---

## FastAPI REST API

`api/` dizininde düzenlenmiş router'lar. Swagger UI: `http://localhost:8001/docs`

| Method | Path | Açıklama |
|--------|------|----------|
| `POST` | `/api/evaluations/run` | Yeni değerlendirme başlat |
| `POST` | `/api/evaluations/runs/{run_id}/cancel` | Çalışan run'ı iptal et |
| `GET` | `/api/evaluations/suites` | Tanımlı test suitlerini listele |
| `GET` | `/api/results/reports` | Raporları listele |
| `GET` | `/api/results/reports/{filename}` | Rapor detayı |
| `GET` | `/api/results/reports/{filename}/raw` | Ham JSON |
| `GET` | `/api/models` | Kayıtlı modelleri listele |
| `POST` | `/api/models` | Yeni model ekle |
| `DELETE` | `/api/models/{key}` | Model sil |
| `GET` | `/api/hitl/pending` | İnceleme bekleyen item'lar |
| `POST` | `/api/hitl/review` | Anotasyon kaydet |
| `GET` | `/api/hitl/stats` | HITL istatistikleri |
| `POST` | `/api/custom-datasets` | Dataset yükle |
| `WS` | `/ws/progress/{run_id}` | Gerçek zamanlı ilerleme akışı |

---

## CI Entegrasyonu (Eval-as-CI)

Değerlendirme çıktısını bir kalite kapısına (gate) dönüştürür: eşikler aşılırsa CI başarısız olur.

### `python -m ci.gate` — CLI

```bash
python -m ci.gate <REPORT.json> [SEÇENEKLER]
```

| Seçenek | Varsayılan | Açıklama |
|---------|-----------|----------|
| `--config PATH` | `config/ci_gate.yaml` | Eşik konfigürasyonu |
| `--baseline PATH` | — | Regresyon karşılaştırması için taban rapor |
| `--format text\|json\|markdown\|badge` | `text` | Çıktı formatı |
| `--output FILE` | stdout | Çıktının yazılacağı dosya |

**Çıkış kodları:**

| Kod | Anlam |
|-----|-------|
| `0` | Gate geçti |
| `1` | Gate başarısız (eşik ihlali) |
| `2` | Kullanım / G/Ç hatası |

**Format notları:**
- `--format markdown` → PR yorumu veya `$GITHUB_STEP_SUMMARY` için tablo
- `--format badge` → shields.io endpoint JSON:
  `{"schemaVersion":1,"label":"llm eval","message":"...","color":"..."}`

### `config/ci_gate.yaml` — Konfigürasyon Anahtarları

```yaml
# Tüm modellerin ağırlıklı skoru bu değerin altına düşerse gate başarısız olur
weighted_score_min: 0.75

# P95 gecikme (saniye) bu değeri aşarsa gate başarısız olur
max_latency_p95: 10.0

# Herhangi bir test hatayla sonuçlanırsa gate başarısız olur
fail_on_test_error: true

# Test bazlı eşikler (isteğe bağlı)
tests:
  turkish_grammar:
    min_score: 0.80
  function_calling:
    min_score: 0.75

# Regresyon eşikleri (--baseline verildiğinde aktif)
regression:
  max_weighted_drop: 0.05   # weighted_score'daki maksimum düşüş
  max_test_drop: 0.10       # tekil test skorundaki maksimum düşüş
```

### pytest Entegrasyonu

```python
# test_eval_gate.py
from ci.pytest_plugin import load_report, assert_weighted_score, assert_test_score, assert_no_regression, assert_gate

def test_quality():
    report = load_report("reports/ci_report.json")
    assert_gate(report)                           # config/ci_gate.yaml kullanır

def test_no_regression():
    report    = load_report("reports/ci_report.json")
    baseline  = load_report("reports/baseline.json")
    assert_no_regression(report, baseline)

def test_grammar_score():
    report = load_report("reports/ci_report.json")
    assert_test_score(report, "turkish_grammar", min_score=0.80)
```

```bash
# Çalıştırma
pytest test_eval_gate.py
```

### GitHub Actions Kullanımı

#### Composite Action (`.github/actions/llm-eval-gate`)

```yaml
- name: LLM Eval Gate
  uses: ./.github/actions/llm-eval-gate
  with:
    report: reports/ci_report.json
    config: config/ci_gate.yaml           # isteğe bağlı
    baseline: reports/baseline.json       # isteğe bağlı, regresyon için
    github-token: ${{ secrets.GITHUB_TOKEN }}  # PR yorumu için
```

Action şunları yapar:
1. Gate'i çalıştırır, Markdown tablosunu `$GITHUB_STEP_SUMMARY`'ye ekler
2. `llm-eval-badge.json` üretir (shields.io endpoint)
3. PR üzerindeyse ve token verilmişse gate özetini PR yorumu olarak gönderir
4. Gate başarısızsa job'u `exit 1` ile bitirir

#### Örnek Workflow

`.github/workflows/llm-eval.yml` dosyası repoya eklenmiştir. `pull_request` ve `workflow_dispatch` tetikleyicilerini destekler. Kullanılan model anahtarlarını ve gerekli API secret'larını kendi ortamınıza göre düzenleyin.

### Shields.io Badge

`llm-eval-badge.json` artifact olarak publish edildikten sonra (örn. GitHub Pages veya release):

```markdown
![llm eval](https://img.shields.io/endpoint?url=<RAW_URL>/llm-eval-badge.json)
```

`<RAW_URL>` yerine `llm-eval-badge.json` dosyasının ham erişim adresini yazın.

---

## İstatistiksel Anlamlılık (Model Karşılaştırma)

Headline `weighted_score` farkı gerçek bir performans farkını mı yansıtıyor, yoksa ölçüm gürültüsü mü? `analysis/significance.py` modülü bu soruyu istatistiksel testlerle yanıtlar; yüzde puanlık bir fark anlamlı olmayabilir.

### CLI Kullanımı

```bash
python -m analysis.significance REPORT.json [--alpha 0.05] [--confidence 0.95] [--seed 42] [--format text|json|markdown] [--output FILE]
```

- `--format markdown` → PR yorumu veya CI step summary olarak doğrudan yapıştırılabilir.
- `--seed` sayesinde bootstrap çıktısı deterministiktir (aynı girdi = aynı sonuç).
- Her zaman exit 0 döner (CI'ı kırmaz).

### Çıktı

**Model bazında bootstrap güven aralığı:**
- Her modelin `per-test overall_score` değerleri üzerinden bootstrap CI
- Modelin `weighted_score`, `n_tests`
- 8'den az testle değerlendirilen model/çift → "small sample" uyarısı

**İkili model karşılaştırması (paired design):**
- Her iki modelin de çalıştırdığı testler üzerinden paired karşılaştırma
- Ortalama fark, paired t-test p-değeri, Wilcoxon signed-rank p-değeri
- Cohen's d_z etki büyüklüğü (negligible / small / medium / large)
- `is_significant` bayrağı (p < alpha) + kazananı bildiren `verdict`

### Python API

```python
from analysis.significance import compute_significance

results = compute_significance("report.json", alpha=0.05, confidence=0.95, seed=42)
# results keys: per_model, pairwise, warnings, alpha, confidence, seed
```

---

## 📦 Eval Datasets

`eval_datasets/` klasörü altındaki tüm dataset dosyaları, farklı LLM yeteneklerini sistematik biçimde test etmek için tasarlanmıştır.

### 🗂️ benchmark/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `turkish_grammar.json` | Türkçe dilbilgisi kuralları: özne-yüklem uyumu, de/da eki, soru eki gibi temel dil hatalarını tespit etme ve düzeltme kapasitesi | `judge_label` (TAM/KISMİ/YANLIŞ), `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `turkish_expression_errors.json` | Anlatım bozukluğu, gereksiz sözcük kullanımı, özne eksikliği ve anlam kayması gibi anlatım hatalarını bulma ve düzeltme | `judge_label`, `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `turkish_creativity.json` | Cümle yeniden yazma, eş anlamlı ifade üretme, metafor kurma, resmi/gündelik dil dönüşümü gibi yaratıcı dil görevleri | `judge_label`, `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `turkish_paraphrasing.json` | Anlam koruyarak farklı sözcüklerle yeniden ifade etme; basit, karmaşık ve aktif/pasif dönüşümleri | `judge_label`, `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `turkish_reasoning.json` | Mantık çıkarımı, matematiksel akıl yürütme, olasılık ve zamanlama hesaplamaları | `reasoning_quality` (LLM judge), `cot_quality` (chain-of-thought), `answer_accuracy` (string match) | 3 metriğin ortalaması |
| `turkish_nuance.json` | Kinaye, deyim, atasözü ve kültürel bağlam yorumlama; gençlik argosu ve söylem nüansları | `judge_label`, `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `language_mix_tests.json` | Türkçe–İngilizce kod değiştirme (code-switching) senaryolarında dil farkındalığı ve yanıt dili doğruluğu | `understanding_rate`, `appropriate_rate`, `avg_consistency`, `score_by_mix_type` | LLM judge ortalaması |
| `turkish_pii_detection.json` | Metinde kişisel veri (PII) varlığını binary olarak tespit etme (0: yok, 1: var); tuzak örneklerle hallucination direnci ölçülür | `accuracy` (binary), `positive_accuracy` (TPR), `negative_accuracy` (TNR), `format_compliance` | `accuracy` |
| `turkish_self_consistency.json` | Aynı soruya farklı yinelemelerde tutarlı yanıt verme yeteneği | `consistency_score`, `overall_similarity`, `temperature_stability`, `stability_rate` | `avg_consistency_score` |
| `negative_constraints_tests.json` | Format (JSON kullanma, liste kullanma), kelime limiti ve yasaklı konu gibi negatif kısıtlamalara uyum | `compliance_rate`, `violation_count`, `severity`, `compliance_by_type` | `compliance_rate` |
| `prompt_compression_tests.json` | %75 / %50 / %25 oranında sıkıştırılmış prompt'larla yanıt kalitesinin bozulup bozulmadığını ölçer | `avg_prompt_reduction` (token azalması), `avg_information_retention`, `avg_quality_score` | `avg_quality_score` |

### 🗂️ agentic/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `agentic_workflows.json` | Çok adımlı görev planlama, uygun araçları seçip sırayla kullanma ve sonucu gerekçeyle sunma (portföy önerisi, harcama analizi vb.) | `plan_quality` (LLM judge ile adım adım plan değerlendirmesi) | `avg_plan_quality` |

### 🗂️ edge_cases/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `adversarial_tests.json` | Prompt injection, jailbreak (DAN modu), rol manipülasyonu ve sosyal mühendislik saldırılarına karşı direnç | `safety_rate`, `refusal_rate`, `compromise_rate`, `safety_by_attack_type`, `risk_distribution` | `safety_rate` |
| `edge_tests.json` | Boş girdi, aşırı uzun metin, emoji/özel karakter ve karmaşık unicode kombinasyonları gibi sınır koşulları | `behavior_score`, `refusal_score` | `avg_behavior_score × (1 - error_rate)` |

### 🗂️ embedding/

Embedding model kalitesini ölçer; üretici LLM değil, vektör temsil gücü değerlendirilir.

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `turkish_sts.json` | Türkçe cümle çiftleri arasındaki anlamsal benzerlik (identical → paraphrase → related → unrelated) | `spearman_correlation`, `pearson_correlation`, `MAE`, `RMSE`, `accuracy_at_threshold` | Spearman korelasyonu |
| `turkish_retrieval.json` | Sorguya doğru dokümanı getirme; pozitif dokümanlar hard negative ve random negative'lerden ayırt edilmeli | `NDCG@1/3/5/10`, `Recall@k`, `Precision@k`, `MRR`, `MAP` | `NDCG@10` |
| `tr_crosslingual_sts.json` | Türkçe–İngilizce cümle çiftlerinin çapraz dil anlamsal benzerliği | `spearman_correlation`, `pearson_correlation`, `MAE`, `RMSE`, `accuracy_at_threshold` | Spearman korelasyonu |
| `tr_hardneg_retrieval.json` | Konu bazında yanıltıcı (hard negative) dokümanların bulunduğu retrieval; anlam ayrımı gerektiren zor örnekler | `NDCG@1/3/5/10`, `Recall@k`, `Precision@k`, `MRR`, `MAP` | `NDCG@10` |
| `fintech_domain.json` | Fintech terminolojisinde anlamsal yakınlık: eş anlamlı terimler kümelenmeli, farklı kavramlar ayrışmalı | `avg_similar_score`, `avg_dissimilar_score`, `separation_margin`, `accuracy`, `pass_rate` | `avg_accuracy` |
| `turkish_regulatory_domain.json` | Bankacılık mevzuatı terimleri (KVKK, BDDK, MASAK, 5411 vb.) için alan-spesifik anlamsal temsil kalitesi | `avg_similar_score`, `avg_dissimilar_score`, `separation_margin`, `accuracy`, `pass_rate` | `avg_accuracy` |

### 🗂️ fintech/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `fintech_knowledge.json` | Bankacılık ve fintech alan bilgisi: EFT/havale farkı, IBAN, KYC, BDDK, lot, asgari ödeme gibi kavramlar | `judge_label`, `judge_score`, `hallucination` | `(TAM×1.0 + KISMİ×0.5) / total` |
| `fintech_calculations.json` | Finansal formüller ve çok adımlı hesaplamalar: eşit taksit, arbitraj fırsatı, getiri oranı, döviz kuru hesabı | `judge_label`, `judge_score`, `hallucination`, sayısal doğruluk (±tolerans) | `(TAM×1.0 + KISMİ×0.5) / total` |

### 🗂️ function_calling/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `function_calling_tests.json` | Kullanıcı isteğine göre doğru aracı seçmek ve parametrelerini eksiksiz doldurmak (temel araç seçimi) | `tool_selection`, `parameter_extraction_lenient`, `parameter_extraction_strict`, `overall_lenient`, `overall_strict` | `overall_lenient` |
| `parallel_tool_tests.json` | Birbirinden bağımsız birden fazla aracın aynı anda çağrılabilmesi (paralel çalışma yeteneği) | `tools_match_rate`, `parallel_detection_rate`, `judge_score` | %40 `tools_match_rate` + %30 `judge_score` + %30 paralel efficiency |
| `tool_chain_tests.json` | Sıralı/koşullu araç zincirleri: bir araç çıktısını sonraki araca girdi olarak kullanma | `tool_coverage` (beklenen araçların kaçı çağrıldı), `order_score` (beklenen sıranın tutulup tutulmadığı) | `0.7 × tool_coverage + 0.3 × order_score` |
| `tool_error_recovery_tests.json` | Araç hataları (timeout, rate limit, geçici arıza) karşısında retry, fallback ve graceful degradation davranışı | `success_rate`, `retry_success_rate`, `fallback_success_rate`, `comprehension_success_rate` | `success_rate` |

### 🗂️ multi_turn/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `conversations.json` | Çok turlu konuşmalarda bağlam hafızası: önceki turdan gelen bilgiyi (bakiye, tercih) doğru hatırlama ve kullanma | `context_retention` (LLM judge, önceki turları hatırlama) | `avg_context_retention` |
| `stress_tests.json` | Uzun konuşma zincirlerinde kümülatif hesaplama ve hafıza zorlaması (gelir/gider takibi, çok adımlı sorular) | `context_retention` (LLM judge) | `avg_context_retention` |

### 🗂️ rag/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `rag_tests.json` | Sağlanan bağlama dayalı yanıt üretme: bağlamda olmayan bilgiyi uydurmama (hallucination), context adherence | `rag_quality`, `context_adherence` | `avg_rag_quality` |
| `needle_in_haystack.json` | Uzun ve alakasız içeriklerle dolu belgelerde gömülü spesifik bir bilgiyi bulma (kritik bilgi erişimi) | `rag_quality`, `context_adherence` | `avg_rag_quality` |

### 🗂️ regression/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `golden.json` | Üretimde baz alınan altın vaka seti: format uyumu, güvenlik refüsü, mahremiyet kuralları ve talimat takibi | `behavior_score`, `refusal_score` | `avg_behavior_score × (1 - error_rate)` |
| `recent_issues.json` | Daha önce raporlanmış hataların (format sapması, hallucination, OTP paylaşımı) tekrar etmediğini doğrulama | `behavior_score`, `refusal_score` | `avg_behavior_score × (1 - error_rate)` |

### 🗂️ security/

| Dosya | Ne test eder? | Metrikler | overall_score |
|-------|--------------|-----------|---------------|
| `security_tests.json` | PII sızıntısı, kimlik doğrulama atlatma, yetki ihlali ve sosyal mühendislik girişimlerine karşı uyum (KVKK, PCI-DSS, ISO 27001) | `behavior_score`, `refusal_score` | `avg_behavior_score × (1 - error_rate)` |
| `stress_tests.json` (security) | Yüksek eş zamanlı istek, büyük payload, derin iç içe JSON ve slow loris gibi sistem düzeyinde stres/saldırı senaryoları | `behavior_score`, `refusal_score` | `avg_behavior_score × (1 - error_rate)` |

---

## Model Konfigürasyonu

`config/models.yaml` şablonu:

```yaml
models:
  gpt-4o:
    provider: openai
    api_key: ${AZURE_OPENAI_KEY}
    api_version: ${AZURE_OPENAI_API_VERSION}
    base_url: ${AZURE_OPENAI_ENDPOINT}
    model_name: ${AZURE_OPENAI_DEPLOYMENT_NAME_PTU}
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true

  qwen-3-30b:
    provider: openai           # vLLM OpenAI-uyumlu
    base_url: ${VLLM_BASE_URL}
    api_key: dummy
    model_name: default
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true

  mistral-small-3.1:
    provider: openai
    base_url: ${MISTRAL_VLLM_BASE_URL}
    api_key: dummy
    model_name: default
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true

  claude-sonnet-45:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    model_name: claude-sonnet-4-20250514
    max_tokens: 4096
    temperature: 0.0
    supports_function_calling: true
```

---

## Ortam Değişkenleri

`.env` dosyası (`.env.example`'dan kopyalayın):

```bash
# Azure OpenAI (judge model + PTU)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME_PTU=gpt-4o
AZURE_OPENAI_DEPLOYMENT_NAME_PR=gpt-4.1

# OpenAI Direct
OPENAI_API_KEY=your-key

# Anthropic
ANTHROPIC_API_KEY=your-key

# On-Premise vLLM (Qwen, Mistral, LLaMA vb.)
VLLM_BASE_URL=http://your-vllm-server:8000/v1
MISTRAL_VLLM_BASE_URL=http://your-mistral-server:8000/v1

# Yerel
OLLAMA_BASE_URL=http://localhost:11434/v1
LMSTUDIO_MODEL1_BASE_URL=http://localhost:1234/v1
```

---

## Çıktı Formatı

Değerlendirme sonuçları `reports/eval_<timestamp>_<hash>.json` olarak kaydedilir:

```json
{
  "timestamp": "2026-02-24T10:30:00",
  "models": {
    "gpt-4o": {
      "model_name": "gpt-4o",
      "provider": "azure",
      "tests": {
        "turkish_grammar": {
          "summary": {
            "total_tests": 15,
            "label_distribution": {
              "TAM_DOGRU": 11, "KISMEN_DOGRU": 3, "YANLIS": 1
            },
            "overall_score": 0.833
          }
        },
        "agentic_workflows": {
          "summary": {
            "agentic_pack": {
              "task_completion": 0.85,
              "tool_correctness": 0.78,
              "response_completeness": 0.82,
              "intent_resolution": 0.79,
              "agentic_pack_aggregate": 0.81
            },
            "overall_score": 0.81
          }
        }
      },
      "overall_metrics": {
        "weighted_score": 0.88,
        "latency_p95": 3.2,
        "error_rate": 0.0,
        "tokens_per_second": 42.5
      }
    }
  },
  "summary": {
    "model_comparison": {
      "gpt-4o": {"weighted_score": 0.88},
      "qwen-3-30b": {"weighted_score": 0.81}
    }
  }
}
```

---

## Proje Yapısı

```
llm-eval-pipeline/
├── api/                         # FastAPI uygulaması
│   ├── main.py                  # App factory, lifespan, CORS
│   ├── config.py                # Pydantic settings
│   ├── routers/                 # Endpoint router'ları
│   └── services/                # İş mantığı (EvalService, HitlService vb.)
├── adapters/
│   ├── unified_adapter.py       # Tüm sağlayıcılar için tekil LLM arayüzü
│   └── embedding_adapter.py     # Embedding modeli arayüzü
├── evaluators/                  # Değerlendirme bileşenleri (bkz. tablo)
├── eval_datasets/               # Test veri setleri (JSON)
│   ├── agentic/
│   ├── benchmark/
│   ├── edge_cases/
│   ├── embedding/
│   ├── fintech/
│   ├── function_calling/
│   ├── multi_turn/
│   ├── rag/
│   ├── regression/
│   └── security/
├── web/                         # React + Vite frontend
│   ├── src/pages/               # Dashboard, Results, HitlReview, RunEvaluation, Models, DatasetStudio
│   ├── src/api/                 # API istemci fonksiyonları
│   ├── src/hooks/               # React hook'ları
│   └── dist/                   # Production build çıktısı
├── config/
│   ├── models.yaml              # Model konfigürasyonları
│   ├── tests.yaml               # Test süitleri + metrik ağırlıkları
│   └── task_registry.yaml       # Task kayıt defteri
├── utils/
│   ├── result_models.py         # Pydantic veri modelleri
│   ├── reproducibility.py       # Tekrarlanabilirlik hash'leme
│   ├── trend_analysis.py        # Tarihsel trend hesaplama
│   ├── cache.py                 # Yanıt önbellekleme
│   └── throughput_metrics.py    # Token/sn throughput hesaplama
├── metrics/                     # Metrik hesaplama yardımcıları
├── pipeline_runner.py           # Ana değerlendirme orkestratörü
├── main.py                      # CLI giriş noktası
├── reports/                     # Değerlendirme sonuçları (JSON)
│   └── evaluations_store.json   # HITL kuyruğu (kalıcı)
├── Makefile
├── docker-compose.yml
├── docker-compose.debug.yml
├── Dockerfile
└── requirements.txt
```

---

## Tekrarlanabilirlik

Her run, `reports/` altına `eval_<timestamp>_<hash>.json` ve yan `.meta.json` dosyası üretir. `config/tests.yaml`'daki `run_seed: 42` parametresi örnek sıralamasını sabitler.

---

## Sorun Giderme

### vLLM Bağlantı Hatası

```
Error: Connection refused to localhost:8000
```

vLLM sunucusunun çalıştığını doğrulayın:

```bash
python -m vllm.entrypoints.openai.api_server --model MODEL_NAME --port 8000
```

### API Key Hatası

```
Error: Invalid API key
```

`.env` dosyasını ve environment variable'ları kontrol edin:

```bash
source .env && echo $AZURE_OPENAI_KEY
```

### Frontend Erişilemiyor

Port 5173 meşgulse Vite otomatik olarak bir sonraki serbest porta geçer; terminal çıktısını kontrol edin.

### Judge Zaman Aşımı

`config/models.yaml`'da timeout değerini artırın:

```yaml
gpt-4o:
  timeout: 60
```

---

## Lisans

MIT License — bkz. [LICENSE](LICENSE)

---

<div align="center">

[⬆ Başa Dön](#llm-evaluation-pipeline)

</div>
