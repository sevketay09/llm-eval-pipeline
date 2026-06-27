#!/bin/bash
# Quick test script for embedding evaluation system

echo "========================================="
echo "Embedding Evaluation System Quick Test"
echo "========================================="
echo ""

# Test 1: LiteLLM API connectivity
echo "Test 1: LiteLLM API Connectivity"
echo "---------------------------------"
python3 << EOF
import httpx
import json
import os

base_url = os.getenv("LITELLM_BASE_URL")
api_key = os.getenv("LITELLM_API_KEY")

if not base_url:
    raise RuntimeError("LITELLM_BASE_URL is not set")

url = base_url.rstrip("/") + "/v1/embeddings"
data = {"model": "jina-v3", "input": ["test"]}
headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

try:
    response = httpx.post(url=url, json=data, headers=headers, timeout=30.0, verify=False)
    response.raise_for_status()
    result = response.json()
    emb_dim = len(result["data"][0]["embedding"])
    print(f"✓ LiteLLM API working. Embedding dimension: {emb_dim}")
except Exception as e:
    print(f"✗ LiteLLM API error: {e}")
EOF

echo ""

# Test 2: Check if sentence-transformers is installed
echo "Test 2: Sentence-Transformers Installation"
echo "-------------------------------------------"
python3 << EOF
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers installed")
except ImportError:
    print("✗ sentence-transformers not installed")
    print("  Install with: pip install sentence-transformers")
EOF

echo ""

# Test 3: Check dataset files
echo "Test 3: Dataset Files"
echo "---------------------"
datasets=(
    "eval_datasets/embedding/turkish_sts.json"
    "eval_datasets/embedding/turkish_retrieval.json"
    "eval_datasets/embedding/fintech_domain.json"
)

for dataset in "${datasets[@]}"; do
    if [ -f "$dataset" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$dataset'))))")
        echo "✓ $dataset ($count items)"
    else
        echo "✗ $dataset not found"
    fi
done

echo ""

# Test 4: Check config files
echo "Test 4: Configuration Files"
echo "----------------------------"
if grep -q "embedding_models:" config/models.yaml; then
    models=$(grep -A 50 "embedding_models:" config/models.yaml | grep "^  [a-z]" | wc -l)
    echo "✓ config/models.yaml has embedding_models section ($models models)"
else
    echo "✗ config/models.yaml missing embedding_models section"
fi

if grep -q "embedding_sts:" config/tests.yaml; then
    echo "✓ config/tests.yaml has embedding test weights"
else
    echo "✗ config/tests.yaml missing embedding test configuration"
fi

echo ""

# Test 5: Run mini embedding evaluation
echo "Test 5: Mini Embedding Evaluation (LiteLLM)"
echo "--------------------------------------------"
python3 << EOF
import json
from adapters.embedding_adapter import UnifiedEmbeddingAdapter
import numpy as np

try:
    # Load config
    import yaml
    with open('config/models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['embedding_models']['jina-v3-litellm']
    model = UnifiedEmbeddingAdapter(model_config, model_key='jina-v3-litellm')
    
    # Test embedding generation
    texts = ["Kredi kartı başvurusu", "Kart başvurusu nasıl yapılır"]
    result = model.encode(texts, normalize=True)
    
    embeddings = result['embeddings']
    
    # Compute similarity
    similarity = np.dot(embeddings[0], embeddings[1])
    
    print(f"✓ Embedding generation successful")
    print(f"  - Embedding dim: {embeddings.shape[1]}")
    print(f"  - Latency: {result['latency']:.3f}s")
    print(f"  - Similarity: {similarity:.4f}")
    
except Exception as e:
    print(f"✗ Embedding test failed: {e}")
EOF

echo ""
echo "========================================="
echo "Quick Test Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run full evaluation:"
echo "     python run_embedding_eval.py --model jina-v3-litellm --test all"
echo ""
echo "  2. Try local HuggingFace model (requires download):"
echo "     python run_embedding_eval.py --model multilingual-e5-large --test sts"
echo ""
echo "  3. View documentation:"
echo "     cat docs/EMBEDDING_EVALUATION.md"
echo ""
