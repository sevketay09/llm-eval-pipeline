# 🤖 LLM Evaluation Pipeline

<div align="center">

**Enterprise-grade evaluation framework for Large Language Models**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Systematically test and compare cloud, on-premise, and local LLMs with comprehensive evaluation metrics*

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 📖 Overview

**LLM Evaluation Pipeline** is a production-ready framework for systematically testing and comparing Large Language Models across multiple dimensions: accuracy, reasoning quality, Turkish language fluency, function calling capabilities, and agentic workflows.

**Who is this for?**
- 🏦 **Enterprises** evaluating LLMs for production deployment
- 🔬 **ML Teams** comparing model performance across providers
- 🧪 **Researchers** benchmarking open-source vs. commercial models
- 💻 **Developers** testing local models (Ollama, LM Studio) before cloud deployment

**Key Benefits:**
- ✅ Unified interface for Azure OpenAI, OpenAI, Anthropic, vLLM, Ollama, LM Studio
- ✅ Comprehensive test suites: Turkish Q&A, reasoning, fintech, function calling, RAG
- ✅ LLM-as-Judge evaluation for nuanced quality assessment
- ✅ Web dashboard for easy model management and result visualization
- ✅ Latency tracking for production readiness
- ✅ Advanced features: Needle in Haystack RAG, Tool Error Recovery, Parallel Tool Execution

---

## 🚀 Quick Start

### Web Dashboard (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

**Browser opens at:** `http://localhost:8501`

**Dashboard features:**
- 🎯 Select multiple models to test
- 🧑‍⚖️ Choose judge model for evaluation
- 📝 Pick test suites or custom tests
- ⚡ Enable parallel execution
- 📊 Real-time progress tracking
- 📈 Result visualization with charts

### Command Line

```bash
# Quick smoke test
python main.py --models gpt4o-azure --suite smoke

# Full evaluation
python main.py --models gpt4o-azure llama3-ollama --suite full
```

### 🐳 Docker (Paylaşım için Önerilen)

```bash
# 1) Environment dosyasını oluştur
cp .env.example .env

# 2) .env içini kendi endpoint/key değerlerinle doldur

# 3) Dashboard'u Docker ile başlat
docker compose up --build
```

Dashboard: `http://localhost:8501`

**Notlar:**
- Repoda key/endpoint saklamayın; tüm hassas değerler `.env` üzerinden okunur.
- `docker-compose.yml` içinde `./config:/app/config` mount edildiği için arayüzden eklenen modeller kalıcı olur.
- Sonuçlar `./reports` altında host üzerinde kalır (container silinse de kaybolmaz).

### 🛠️ Makefile ile Debug Akışı

```bash
# Hızlı başlangıç (down + build + up)
make start-debug

# Sadece ayağa kaldır
make up-debug

# Logları izle
make tail-logs

# Durdur
make down-debug
```

**Base image değiştirme (opsiyonel):**

```bash
make start-debug BASE_IMAGE=your-registry/your-image:tag
```

---

## ✨ Features

###  Evaluation Metrics

**Accuracy Metrics:**
- Exact match, fuzzy match, semantic similarity
- Numerical accuracy for calculations
- Tool selection accuracy
- Parameter extraction correctness

**LLM-as-Judge (Categorical):**
- **TAM_DOGRU** (1.0): Answer is fully correct and complete
- **KISMEN_DOGRU** (0.5): Answer is partially correct or incomplete
- **YANLIS** (0.0): Answer is incorrect, irrelevant, or hallucinated
- **Overall Score**: `(TAM_DOGRU × 1.0 + KISMEN_DOGRU × 0.5) / total`

The judge outputs structured JSON `{"label": "TAM_DOGRU|KISMEN_DOGRU|YANLIS", "reasoning": "..."}` — no numeric 1-10 scale, no verbosity bias.

**Performance Metrics:**
- Latency: p50, p95, p99 percentiles
- Throughput: tokens per second

**Per-Category Breakdown:**
- TAM_DOGRU / KISMEN_DOGRU / YANLIS counts per category
- `tam_dogru_rate` per category for quick comparison

**Safety:**
- Prompt injection resistance
- Hallucination detection
- Refusal handling

### 🚀 Advanced Features

#### 1. Needle in Haystack RAG Evaluation
Tests model's ability to find specific information buried in long context documents.

```python
# Automatically tests retrieval accuracy with varying context lengths
# Measures: retrieval precision, context utilization, distractor resistance
```

#### 2. Tool Error Recovery
Evaluates how models handle tool failures and retry logic.

```python
# Simulates API failures, timeout errors, validation issues
# Measures: recovery strategies, retry logic, graceful degradation
```

#### 3. Parallel Tool Execution
Tests models that can call multiple tools simultaneously for efficiency.

```python
# Checks parallel tool calling capability
# Measures: parallelization accuracy, coordination, result aggregation
```

### 🎯 Advanced Evaluation Capabilities

#### 4. Turkish PII Detection
Evaluates model's ability to identify and handle Turkish Personally Identifiable Information (PII).

```python
# Tests detection of: TC Kimlik No, phone numbers, IBAN, email, address
# Measures: detection accuracy, false positives, masking quality
# Dataset: 15 test cases covering various PII scenarios
```

**Configuration:**
```yaml
metric_weights:
  pii_detection: 0.15  # Weighted into overall score
```

#### 5. Self-Consistency Evaluation
Tests model's consistency across multiple generations for the same prompt.

```python
# Runs same prompt N times (default: 5)
# Measures: semantic consistency, consensus quality, variance
# Applies majority voting for improved accuracy
```

**Usage:**
```bash
# Tests automatically run with consistency checking
# Results include consistency_score in metrics
```

#### 6. Human-in-the-Loop Review
Interactive web interface for manual evaluation and quality control.

```bash
# Launch HITL dashboard
streamlit run dashboard.py
```

**Features:**
- 🔍 Manual review of model responses
- ✅ Approve/reject decisions with categorical labels (TAM DOĞRU / KISMİ DOĞRU / YANLIŞ)
- 📊 Side-by-side model comparisons
- 💾 Export annotated datasets
- 📈 Quality trend tracking

**Dashboard Pages:**
- **Review Results:** Confirm or override judge label (TAM DOĞRU / KISMİ DOĞRU / YANLIŞ), add comments, flag issues
- **Comparative Analysis:** Compare models side-by-side, highlight differences

#### 7. Prompt Compression Testing
Evaluates model performance with compressed prompts to reduce token usage.

```python
# Compression techniques: 
# - remove_stopwords: Removes Turkish stopwords
# - extract_keywords: Keeps only keywords
# - summarize: Condenses to core meaning
# Measures: accuracy retention, compression ratio
```

**Results format:**
```json
{
  "original": {"accuracy": 0.85, "tokens": 150},
  "compressed": {"accuracy": 0.82, "tokens": 45},
  "compression_ratio": 3.3,
  "accuracy_retention": 0.965
}
```

#### 8. Negative Constraints Testing
Tests model's ability to follow prohibitions and restrictions.

```python
# Tests constraints like:
# - "Do not mention X"
# - "Avoid using Y words"
# - "Keep under N words"
# Measures: violation detection, constraint adherence, edge case handling
```

**Example tests:**
- Forbidden word usage
- Length constraints
- Topic restrictions
- Format compliance

#### 9. Adversarial & Jailbreak Testing
Evaluates model robustness against prompt injection and jailbreak attempts.

```python
# Test scenarios:
# - Prompt injection attacks
# - Role manipulation attempts
# - Instruction override attempts
# - Indirect jailbreaks
# Measures: refusal rate, safety compliance, boundary enforcement
```

**Safety categories tested:**
- Role confusion
- Instruction override
- Encoding tricks (Base64, ROT13)
- Indirect attacks

#### 10. Language Mix Testing (Turkish-English)
Evaluates model handling of code-switching and mixed-language scenarios.

```python
# Tests scenarios:
# - Turkish question → English answer requirement
# - English terms in Turkish context
# - Technical documentation mixing
# - Natural code-switching in conversations
# Measures: language understanding, instruction following, mixing quality
```

**Test categories:**
- Turkish to English translation
- English to Turkish translation
- Mixed-language reasoning
- Code-switching detection
- Technical term handling

**Configuration:**
```yaml
language_mix:
  weight: 0.10
  enable_detection: true
  penalty_for_wrong_language: 0.5
```

### 🤖 Supported Providers

| Provider | Type | API Compatibility | Use Case |
|----------|------|-------------------|----------|
| **Azure OpenAI** | Cloud | Native | Enterprise production |
| **OpenAI** | Cloud | Native | General purpose |
| **Anthropic** | Cloud | Native | Advanced reasoning |
| **vLLM** | On-Premise | OpenAI-compatible | Self-hosted deployment |
| **Ollama** | Local | OpenAI-compatible | Development & testing |
| **LM Studio** | Local | OpenAI-compatible | Desktop testing |

---

## � Eval Datasets

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

## �📦 Installation

### Prerequisites

- **Python 3.12+**
- **pip** or **conda**
- (Optional) **Ollama** for local models
- (Optional) **LM Studio** for GGUF models

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/llm-eval-pipeline.git
cd llm-eval-pipeline
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `streamlit>=1.29.0` - Web dashboard
- `openai>=1.0.0` - OpenAI/Azure/vLLM/Ollama/LM Studio client
- `anthropic>=0.18.0` - Anthropic Claude client
- `pyyaml` - Configuration files
- `pandas` - Data processing
- `plotly` - Visualization

### Step 3: Configure Environment Variables (Optional)

For cloud providers, create `.env` file:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here

# OpenAI Direct
OPENAI_API_KEY=your-key-here

# Anthropic
ANTHROPIC_API_KEY=your-key-here

# vLLM On-Premise
LLM_API_URL=http://your-vllm-server:8000/v1
```

### Step 4: Verify Installation

```bash
# Quick smoke test with default config
python main.py --models gpt4o-azure --suite smoke
```

---

## 💻 Usage

### 🌐 Web Dashboard (Recommended)

Launch the Streamlit dashboard for interactive model testing:

```bash
streamlit run app.py
```

**Dashboard Tabs:**

#### 1️⃣ **Run Evaluation**
- Select models (multi-select by provider)
- Choose judge model for LLM-as-Judge evaluation
- Pick test suite or customize tests
- Enable parallel execution
- Monitor real-time progress

**Steps:**
1. Select at least 1 model to test
2. Select judge model (e.g., GPT-4o)
3. Choose test suite: `smoke`, `full`, `fintech_only`, etc.
4. Click "EVALUATION BAŞLAT" (Start Evaluation)
5. Monitor progress bar and logs

#### 2️⃣ **View Results**
- Browse evaluation reports
- Compare models side-by-side
- Interactive charts (bar, radar, line)
- Export results to CSV

#### 3️⃣ **Configuration**
- Add/delete models
- View test suite definitions
- Edit YAML configurations

**Adding Models via Dashboard:**
1. Go to "Configuration" tab
2. Select provider (Azure OpenAI, OpenAI, Anthropic, vLLM, Ollama, LM Studio)
3. Fill in model details (name, endpoint, API key)
4. Click "Model Ekle" (Add Model)
5. Model is immediately available for testing

### 🔍 Human-in-the-Loop Review Dashboard

For manual evaluation and quality control:

```bash
streamlit run dashboard.py
```

**Dashboard Pages:**

#### 1️⃣ **Review Results**
- Load evaluation reports (JSON files from `reports/` folder)
- Review LLM judge decision (✅ TAM DOĞRU / 🟡 KISMİ DOĞRU / ❌ YANLIŞ)
- Confirm or override label with manual annotation
- Add detailed comments and flag problematic responses
- Track review progress

**Review Features:**
- Test-by-test navigation
- Expected vs. actual answer comparison
- LLM judge label + reasoning shown for context
- Notes field for feedback
- Export annotated results as training data

#### 2️⃣ **Comparative Analysis**
- Select two models for side-by-side comparison
- Highlight response differences
- Vote for better response
- Track win/loss statistics
- Export comparison report

**HITL Queue (pending_reviews.jsonl):**
- Each item includes `llm_judge_label` (TAM_DOGRU / KISMEN_DOGRU / YANLIS) from the judge
- Reviewers confirm or override the label
- Annotations exported as training data

**Use Cases:**
- Quality assurance before production deployment
- Fine-tuning dataset creation from LLM+human annotations
- Edge case identification and documentation
- Model A/B testing with human judgment

### 💻 Command Line Interface

#### Basic Usage

```bash
python main.py --models MODEL_NAME [OPTIONS]
```

**Options:**
- `--models`: Model(s) to test (comma-separated or space-separated)
- `--suite`: Test suite name (see [Test Suites](#-test-suites))
- `--output`: Output JSON file path (default: `reports/eval_<timestamp>.json`)
- `--parallel`: Run models in parallel (faster)

#### Examples

**Single Model Test:**
```bash
python main.py --models gpt4o-azure --suite smoke
```

**Compare Multiple Models:**
```bash
python main.py --models gpt4o-azure claude-sonnet-45 llama3-ollama --suite full
```

**Fintech-Only Evaluation:**
```bash
python main.py --models mistral-small-onprem --suite fintech_only
```

**Custom Output Path:**
```bash
python main.py --models gpt4o-azure --suite full --output reports/production_test_$(date +%Y%m%d).json
```

**Parallel Execution:**
```bash
python main.py --models model1 model2 model3 --suite full --parallel
```

### 📊 Output Format

Evaluation results are saved as JSON:

```json
{
  "timestamp": "2026-02-24T10:30:00",
  "models": {
    "gpt4o-azure": {
      "model_name": "gpt-4o",
      "provider": "azure",
      "tests": {
        "turkish_qa": {
          "summary": {
            "total_tests": 15,
            "label_distribution": {
              "TAM_DOGRU": 11, "KISMEN_DOGRU": 3, "YANLIS": 1,
              "tam_dogru_rate": 0.733, "kismen_dogru_rate": 0.200, "yanlis_rate": 0.067
            },
            "avg_scores": {
              "tam_dogru_rate": 0.733,
              "kismen_dogru_rate": 0.200,
              "yanlis_rate": 0.067,
              "avg_hallucination": 0.94
            },
            "category_breakdown": {
              "banking_jargon": {"count": 4, "TAM_DOGRU": 3, "KISMEN_DOGRU": 1, "YANLIS": 0, "tam_dogru_rate": 0.75}
            },
            "avg_latency": 1.2,
            "overall_score": 0.833
          },
          "results": [{}, ...]
        }
      },
      "overall_metrics": {
        "weighted_score": 0.88,
        "latency_p95": 3.2
      }
    }
  },
  "summary": {
    "model_comparison": {
      "gpt4o-azure": {"overall_score": 0.88},
      "llama3-ollama": {"overall_score": 0.76}
    },
    "best_performers": {...}
  }
}
```

---

## 🤖 Model Configuration

### Adding New Models

Edit [config/models.yaml](config/models.yaml) or use the dashboard Configuration tab.

### Cloud Providers

#### Azure OpenAI

```yaml
gpt4o-azure:
  provider: azure
  azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  model_name: gpt-4o
  api_version: "2024-08-01-preview"
  supports_function_calling: true
  temperature: 0.0
  max_tokens: 4096
```

**Environment variables:**
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
```

#### OpenAI Direct

```yaml
gpt4o-openai:
  provider: openai
  api_key: ${OPENAI_API_KEY}
  model_name: gpt-4o
  supports_function_calling: true
  temperature: 0.0
  max_tokens: 4096
```

#### Anthropic Claude

```yaml
claude-sonnet-45:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}
  model_name: claude-sonnet-4-20250514
  supports_function_calling: true
  temperature: 0.0
  max_tokens: 4096
```

### On-Premise (vLLM)

```yaml
qwen-72b-vllm:
  provider: openai  # vLLM is OpenAI-compatible
  base_url: http://your-vllm-server:8000/v1
  api_key: dummy  # vLLM doesn't require real key
  model_name: Qwen/Qwen2.5-72B-Instruct
  supports_function_calling: true
  temperature: 0.0
  max_tokens: 4096
```

**Start vLLM server:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4
```

### Local Models

#### Ollama

```yaml
llama3-ollama:
  provider: openai  # Ollama is OpenAI-compatible
  base_url: http://localhost:11434/v1
  api_key: dummy
  model_name: llama3
  supports_function_calling: false
  temperature: 0.0
  max_tokens: 4096
```

**Setup Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (default port: 11434)
ollama serve

# Pull and run model
ollama pull llama3
ollama run llama3
```

**Available Ollama models:** `llama3`, `llama3.1`, `mistral`, `qwen2.5`, `gemma2`, `phi3`

#### LM Studio

```yaml
lmstudio-model1:
  provider: openai  # LM Studio is OpenAI-compatible
  base_url: http://localhost:1234/v1
  api_key: dummy
  model_name: local-model
  supports_function_calling: false
  temperature: 0.0
  max_tokens: 4096
```

**Setup LM Studio:**
1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Load GGUF model in UI
3. Go to "Server" tab
4. Click "Start Server" (default port: 1234)
5. Model is accessible at `http://localhost:1234/v1`

**Multiple LM Studio instances:** Run different models on different ports (1234, 1235, etc.)

---

## 🏗️ Architecture

### Project Structure

```
llm-eval-pipeline/
├── main.py                      # CLI entry point
├── app.py                       # Streamlit evaluation dashboard
├── dashboard.py                 # Human-in-the-loop review dashboard
├── pipeline_runner.py           # Core evaluation logic
├── adapters/
│   └── unified_adapter.py       # Unified LLM interface
├── evaluators/
│   ├── accuracy_eval.py         # Accuracy metrics (fallback for no expected_answer)
│   ├── llm_judge.py             # LLM-as-Judge: categorical TAM_DOGRU/KISMEN_DOGRU/YANLIS
│   ├── safety_eval.py           # Safety checks (PII, adversarial)
│   ├── advanced_eval.py         # Needle, Tool Error, Parallel Tools
│   ├── consistency_eval.py      # Self-consistency evaluation
│   ├── benchmark_eval.py        # Prompt compression, constraints
│   ├── comparative_eval.py      # Model comparison utilities
│   └── language_mix_eval.py     # Turkish-English mix testing
├── eval_datasets/
│   ├── benchmark/               # Turkish Q&A, reasoning, language mix
│   ├── fintech/                 # Finance domain tests
│   ├── function_calling/        # Tool usage tests
│   ├── agentic/                 # Multi-step workflows
│   ├── multi_turn/              # Multi-turn conversations
│   └── rag/                     # RAG tests
├── config/
│   ├── models.yaml              # Model configurations
│   └── tests.yaml               # Test suite definitions
├── docs/
│   └── openapi.yaml             # OpenAPI 3.0 REST API specification
├── reports/                     # Evaluation results (JSON)
└── utils/
    ├── cache.py                 # Response caching
    ├── structured_output.py     # JSON schema enforcement
    ├── trend_analysis.py        # Historical comparison
    └── schema_registry.py       # Schema validation
```

### Core Components

#### 1. UnifiedAdapter

Provides single interface for all LLM providers:

```python
from adapters.unified_adapter import UnifiedAdapter

adapter = UnifiedAdapter(model_config)
response = adapter.generate(prompt, temperature=0.0)
```

Supports: Azure OpenAI, OpenAI, Anthropic, vLLM, Ollama, LM Studio

#### 2. PipelineRunner

Orchestrates evaluation workflow:

1. Load test datasets
2. Run tests for each model
3. Evaluate responses (accuracy + LLM-as-Judge)
4. Aggregate scores
5. Generate reports

#### 3. Evaluators

**AccuracyEvaluator:** Exact match, fuzzy match, numerical accuracy  
**LLMJudge:** Categorical evaluation — TAM_DOGRU / KISMEN_DOGRU / YANLIS labels with reasoning  
**SafetyEvaluator:** Prompt injection, hallucination detection  
**AdvancedEvaluator:** Needle RAG, Tool Error Recovery, Parallel Tools

### Advanced Features Implementation

#### Needle in Haystack RAG

Tests retrieval accuracy with varying context lengths:

```python
# Embeds docs: 1K, 10K, 100K tokens
# Inserts "needle" (key information) at random positions
# Measures: retrieval precision, distractor resistance
```

#### Tool Error Recovery

Simulates tool failures and evaluates retry logic:

```python
# Injects: API failures, timeouts, validation errors
# Measures: recovery strategy quality, retry attempts, graceful degradation
```

#### Parallel Tool Execution

Tests simultaneous tool calling:

```python
# Requires calling multiple tools in parallel
# Measures: parallelization accuracy, result aggregation
```

---

## � API Documentation

### OpenAPI Specification

Complete REST API documentation available in OpenAPI 3.0 format:

📄 **[docs/openapi.yaml](docs/openapi.yaml)**

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/models` | POST | Add new model |
| `/tests/suites` | GET | List test suites |
| `/evaluations` | POST | Start evaluation |
| `/evaluations/{id}` | GET | Get evaluation status |
| `/evaluations/{id}/results` | GET | Get evaluation results |
| `/evaluations/{id}` | DELETE | Cancel evaluation |
| `/reports` | GET | List all reports |
| `/reports/{id}` | GET | Get specific report |

**Example: Start Evaluation**

```bash
curl -X POST http://localhost:8000/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["gpt4o-azure", "llama3-ollama"],
    "suite": "smoke",
    "judge_model": "gpt4o-azure",
    "parallel": true
  }'
```

**Response:**
```json
{
  "evaluation_id": "eval_20260224_103045",
  "status": "running",
  "models": ["gpt4o-azure", "llama3-ollama"],
  "suite": "smoke",
  "created_at": "2026-02-24T10:30:45Z"
}
```

**Viewing API Documentation:**

```bash
# Using Swagger UI (requires swagger-ui installation)
# or import docs/openapi.yaml into Postman, Insomnia, etc.
```

**Interactive API Testing:**

1. Import [docs/openapi.yaml](docs/openapi.yaml) into:
   - **Postman:** File → Import → Select openapi.yaml
   - **Insomnia:** Create → Import → Select openapi.yaml
   - **Swagger Editor:** [editor.swagger.io](https://editor.swagger.io/)

2. Configure base URL: `http://localhost:8000/api/v1`

3. Test endpoints interactively with auto-generated request/response examples

**API Schemas:**

All request/response schemas are fully documented in the OpenAPI spec:
- `EvaluationRequest`: Start evaluation parameters
- `EvaluationResponse`: Evaluation results format
- `ModelConfig`: Model configuration schema
- `TestSuite`: Test suite definition schema
- `ErrorResponse`: Standard error format

---

## �🛠️ Development Guide

### Adding New Tests

1. **Create dataset:**

```bash
# eval_datasets/custom/my_new_test.json
[
  {
    "id": "test_1",
    "input": "Test question",
    "expected_output": "Expected answer",
    "category": "custom"
  }
]
```

2. **Add test function in `pipeline_runner.py`:**

```python
def run_my_new_test(self, model_name, max_samples=None):
    tests = self.load_dataset("custom/my_new_test.json")
    results = []
    for test in tests[:max_samples]:
        response = self.adapters[model_name].generate(test["input"])
        score = self.evaluate_accuracy(response, test["expected_output"])
        results.append({"test_id": test["id"], "score": score})
    return results
```

3. **Add to test suite in `config/tests.yaml`:**

```yaml
suites:
  my_suite:
    description: "My custom suite"
    tests:
      - my_new_test
    max_samples_per_test: 10
```

### Adding New Evaluators

Create new evaluator in `evaluators/`:

```python
# evaluators/my_evaluator.py
class MyEvaluator:
    def evaluate(self, response, expected):
        # Your evaluation logic
        return score
```

Import and use in `pipeline_runner.py`:

```python
from evaluators.my_evaluator import MyEvaluator

evaluator = MyEvaluator()
score = evaluator.evaluate(response, expected)
```

### Scoring System

The pipeline uses a **categorical label system** — no numeric 1-10 scale:

| Label | Score | Meaning |
|-------|-------|---------|
| `TAM_DOGRU` | 1.0 | Fully correct |
| `KISMEN_DOGRU` | 0.5 | Partially correct |
| `YANLIS` | 0.0 | Wrong / irrelevant |

**Overall score per test:**
```
overall_score = (TAM_DOGRU × 1.0 + KISMEN_DOGRU × 0.5) / total
```

Per-category breakdown (`category_breakdown` in results):
```json
{
  "banking_jargon": {
    "count": 4,
    "TAM_DOGRU": 3,
    "KISMEN_DOGRU": 1,
    "YANLIS": 0,
    "tam_dogru_rate": 0.75
  }
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### vLLM Connection Error
```
Error: Connection refused to localhost:8000
```
**Solution:** Verify vLLM server is running:
```bash
python -m vllm.entrypoints.openai.api_server --model MODEL_NAME --port 8000
```

#### API Key Error
```
Error: Invalid API key
```
**Solution:** Check `.env` file and ensure variables are set correctly:
```bash
source .env
echo $AZURE_OPENAI_API_KEY  # Should print your key
```

#### Out of Memory
```
Error: CUDA out of memory
```
**Solution:**
- Increase `--tensor-parallel-size` in vLLM
- Reduce batch size
- Use smaller model or quantized version

#### Judge Model Timeout
```
Error: Request timeout after 30s
```
**Solution:** Increase timeout in `config/models.yaml`:
```yaml
gpt4o-ptu:
  timeout: 60  # Increase to 60 seconds
```

#### Ollama/LM Studio Connection Issues

**Ollama not responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**LM Studio model not loaded:**
1. Open LM Studio UI
2. Verify model is loaded in "Server" tab
3. Check port number matches `config/models.yaml`

**Port conflicts:**
```bash
# Check if port is in use
lsof -i :11434  # Ollama default
lsof -i :1234   # LM Studio default

# Use different port
ollama serve --port 11435
```

---

## 🎯 Example Use Cases

### Scenario 1: Evaluating a New Model

```bash
# 1. Quick smoke test
python main.py --models new-model --suite smoke

# 2. If passed, run full evaluation
python main.py --models new-model --suite full

# 3. Compare with production model
python main.py --models gpt4o-azure new-model --suite full
```

### Scenario 2: Testing Local Models

```bash
# 1. Start Ollama
ollama serve
ollama run llama3

# 2. Test locally before deploying
python main.py --models llama3-ollama --suite full_local

# 3. Compare with cloud baseline
python main.py --models llama3-ollama gpt4o-azure --suite full
```

### Scenario 3: Fintech Deployment

```bash
# 1. Test fintech-specific capabilities
python main.py --models candidate-model --suite fintech_only

# 2. Verify function calling
python main.py --models candidate-model --suite function_calling

# 3. Full evaluation if passed
python main.py --models candidate-model --suite full
```

### Scenario 4: CI/CD Integration

```yaml
# .github/workflows/model-eval.yml
name: Model Evaluation
on:
  schedule:
    - cron: '0 2 * * 1'  # Every Monday 2am
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run evaluation
        run: python main.py --models gpt4o-azure --suite full
      - name: Check thresholds
        run: python utils/check_thresholds.py reports/eval_*.json
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/my-new-feature`
3. **Make changes:**
   - Add tests for new features
   - Update documentation
   - Follow code style (black, flake8)
4. **Test thoroughly:** Run `python main.py --models MODEL --suite smoke`
5. **Submit pull request**

### Code Style

```bash
# Format code
black .

# Lint
flake8 --max-line-length=120

# Type checking
mypy --ignore-missing-imports .
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web dashboard framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) - Universal LLM client
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Claude integration
- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference
- [Ollama](https://ollama.com/) - Local model runtime
- [LM Studio](https://lmstudio.ai/) - GGUF model deployment
- [langdetect](https://github.com/Mimino666/langdetect) - Language detection for mix testing
- [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy string matching
- [pyyaml](https://pyyaml.org/) - Configuration management
- [pandas](https://pandas.pydata.org/) - Data processing
- [plotly](https://plotly.com/) - Interactive visualizations

---

<div align="center">

[⬆ Back to Top](#-llm-evaluation-pipeline)

</div>
