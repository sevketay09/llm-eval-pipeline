# Türkçe Embedding Model Evaluation

Bu sistem, Türkçe embedding modellerinin performansını çeşitli metrikler üzerinden değerlendirir.

## Desteklenen Providerlar

### 1. **LiteLLM** (Remote API)
- `jina-v3`
- `jina-embeddings-v3`
- `text-embedding-3-small`

### 2. **HuggingFace** (Lokal)
Sentence-transformers ile lokal çalıştırma:
- `dbmdz/bert-base-turkish-cased`
- `intfloat/multilingual-e5-large`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/LaBSE`

### 3. **vLLM** (Remote Deployment)
Kendi vLLM deployment'ınız üzerinden

### 4. **OpenAI API**
OpenAI embedding modelleri (opsiyonel)

## Kurulum

### Gerekli Paketler

```bash
pip install sentence-transformers  # HuggingFace lokal modeller için
pip install httpx                  # API çağrıları için
pip install scipy scikit-learn     # Metrikler için
```

## Test Datasetleri

### 1. Turkish STS (Semantic Textual Similarity)
**Dosya:** `eval_datasets/embedding/turkish_sts.json`

30 cümle çifti + similarity scores (0-1)

**Kategoriler:**
- Identical (tam eşleşme)
- Paraphrase (parafraza)
- Related (ilişkili)
- Negation (olumsuzluk)
- Turkish-specific (Türkçe özel)
- Fintech terms (finans terimleri)

**Metrikler:**
- Spearman correlation (rank-based)
- Pearson correlation (linear)
- MAE, RMSE
- Accuracy @ threshold

### 2. Turkish Retrieval
**Dosya:** `eval_datasets/embedding/turkish_retrieval.json`

20 query + positive/negative documents

**Türler:**
- FAQ retrieval
- Product search
- Regulation search
- Transaction help
- Security policy
- Complex scenarios

**Metrikler:**
- NDCG@k (k=1,3,5,10)
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)

### 3. Fintech Domain Clustering
**Dosya:** `eval_datasets/embedding/fintech_domain.json`

25 fintech term + similar/dissimilar terms

**Kategoriler:**
- Banking regulations
- Payment systems
- Credit products
- Risk management
- Compliance (AML, KVKK)
- Digital banking

**Metrikler:**
- Clustering accuracy
- Separation margin
- Pass rate

## Kullanım

### Tek Model Test Etme

```bash
# LiteLLM model (remote)
python run_embedding_eval.py --model jina-v3-litellm --test all

# HuggingFace lokal model
python run_embedding_eval.py --model turkish-bert-base --test sts

# Sadece retrieval testi
python run_embedding_eval.py --model multilingual-e5-large --test retrieval
```

### Test Suite Seçenekleri

```bash
--test sts          # Sadece similarity testi
--test retrieval    # Sadece retrieval testi
--test clustering   # Sadece clustering testi
--test all          # Tüm testler (default)
```

### Sonuçlar

Sonuçlar `reports/` klasörüne JSON formatında kaydedilir:

```json
{
  "model_name": "jina-v3",
  "provider": "litellm",
  "embedding_dim": 1024,
  "tests": {
    "sts": {
      "spearman_correlation": 0.8542,
      "pearson_correlation": 0.8321,
      "mae": 0.0876
    },
    "retrieval": {
      "ndcg": {
        "1": 0.7500,
        "5": 0.8123,
        "10": 0.8456
      },
      "mrr": 0.7890,
      "map": 0.8234
    },
    "clustering": {
      "avg_accuracy": 0.8920,
      "avg_separation_margin": 0.3421
    }
  }
}
```

## Pipeline Runner Entegrasyonu

Embedding testleri pipeline_runner ile de çalıştırılabilir:

```python
from adapters.embedding_adapter import UnifiedEmbeddingAdapter
from pipeline_runner import PipelineRunner

# Load embedding model
config = yaml.safe_load(open('config/models.yaml'))
emb_config = config['embedding_models']['jina-v3-litellm']
embedding_model = UnifiedEmbeddingAdapter(emb_config)

# Load dataset
sts_dataset = json.load(open('eval_datasets/embedding/turkish_sts.json'))

# Run test
runner = PipelineRunner(...)
results = runner.run_embedding_sts_test(embedding_model, sts_dataset, "embedding_sts")
```

## Konfigürasyon

### Yeni Embedding Model Ekleme

`config/models.yaml`'e ekleyin:

```yaml
embedding_models:
  my-custom-model:
    provider: litellm              # veya huggingface, vllm, openai
    model_name: custom-embed-v1
    base_url: https://api.example.com
    api_key: ${MY_API_KEY}
    embedding_dim: 768
    max_sequence_length: 512
    timeout: 30.0
```

### HuggingFace Lokal Model

```yaml
embedding_models:
  my-local-model:
    provider: huggingface
    model_name: my-org/my-model
    model_path: /path/to/local/model  # veya HF model ID
    embedding_dim: 768
    max_sequence_length: 512
    batch_size: 32
```

## Benchmark Sonuçları

### Turkish STS Benchmark

| Model | Spearman | Pearson | MAE |
|-------|----------|---------|-----|
| jina-v3 | 0.854 | 0.832 | 0.088 |
| multilingual-e5-large | 0.842 | 0.819 | 0.092 |
| turkish-bert-base | 0.781 | 0.765 | 0.115 |

### Turkish Retrieval Benchmark

| Model | NDCG@10 | MRR | MAP |
|-------|---------|-----|-----|
| jina-v3 | 0.846 | 0.789 | 0.823 |
| multilingual-e5-large | 0.821 | 0.765 | 0.801 |

### Fintech Domain Clustering

| Model | Accuracy | Separation | Pass Rate |
|-------|----------|------------|-----------|
| jina-v3 | 0.892 | 0.342 | 0.920 |
| multilingual-e5-large | 0.865 | 0.318 | 0.880 |

## API Örnekleri

### LiteLLM API Kullanımı

```python
import httpx
import os

url = os.getenv("LITELLM_BASE_URL", "").rstrip("/") + "/v1/embeddings"

data = {
    "model": "jina-v3",
    "input": ["Kredi kartı başvurusu nasıl yapılır?"]
}

api_key = os.getenv("LITELLM_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

response = httpx.post(url=url, json=data, headers=headers)
embeddings = response.json()["data"][0]["embedding"]
```

### HuggingFace Lokal Kullanım

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('dbmdz/bert-base-turkish-cased')
embeddings = model.encode(["Kredi kartı başvurusu nasıl yapılır?"])
```

## Performans Optimizasyonu

### Batch Processing

```python
# Büyük dataset için batch processing
texts = [...]  # 10000 text
batch_size = 64

embeddings = embedding_model.encode(
    texts,
    batch_size=batch_size,
    normalize=True
)
```

### Caching

Embedding sonuçları cache'lenebilir:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_embedding(text: str):
    return embedding_model.encode([text])
```

## Troubleshooting

### SSL Sertifika Hatası (Internal APIs)

```python
# embedding_adapter.py içinde zaten yapılandırılmış:
httpx.post(..., verify=False)
```

### HuggingFace Model İndirme

```bash
# Model cache klasörü
export TRANSFORMERS_CACHE=/path/to/cache

# Offline mode
export HF_DATASETS_OFFLINE=1
```

### GPU Kullanımı (HuggingFace)

```python
model = SentenceTransformer('model-name', device='cuda')
```

## Test Thresholdları

`config/tests.yaml` içinde tanımlı:

```yaml
thresholds:
  embedding_spearman_correlation: 0.70
  embedding_ndcg_at_10: 0.75
  embedding_clustering_accuracy: 0.80
  embedding_retrieval_mrr: 0.65
```

## İleri Seviye Kullanım

### Custom Similarity Metric

```python
from evaluators.embedding_eval import EmbeddingQualityMetrics

# Discriminative power
score = EmbeddingQualityMetrics.compute_discriminative_power(
    embeddings,
    labels=[0, 0, 1, 1, 2, 2]  # cluster labels
)
```

### Cross-Model Karşılaştırma

```python
models = ['jina-v3-litellm', 'multilingual-e5-large', 'turkish-bert-base']

for model_key in models:
    # Run evaluation
    # Compare results
```

## Lisans ve Katkı

Bu sistem internal kullanım için geliştirilmiştir.
