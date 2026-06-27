#!/bin/bash
# Clear all cache and old test results

echo "🧹 Temizlik başlıyor..."

# Python bytecode cache
echo "  → Python cache temizleniyor..."
find . -path ./venv -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -path ./venv -prune -o -type f -name "*.pyc" -delete 2>/dev/null

# Eval results cache
if [ -d ".cache/eval_results" ]; then
    echo "  → Eval results cache temizleniyor..."
    rm -rf .cache/eval_results/*
fi

# Eski eval_*.json rapor dosyaları temizleniyor
if ls reports/eval_*.json 2>/dev/null | grep -q .; then
    echo "  → Eski eval_*.json dosyaları temizleniyor..."
    rm -f reports/eval_*.json
fi

# Evaluation store temizleniyor
if [ -f "reports/evaluations_store.json" ]; then
    echo "  → Evaluation store temizleniyor..."
    rm -f reports/evaluations_store.json
fi

# Pending annotations temizleniyor
if [ -d "annotations/pending" ]; then
    echo "  → Pending annotations temizleniyor..."
    rm -f annotations/pending/* 2>/dev/null || true
fi

# Completed annotations temizleniyor
if [ -d "annotations/completed" ]; then
    echo "  → Completed annotations temizleniyor..."
    rm -f annotations/completed/*.json 2>/dev/null || true
fi

echo "✅ Temizlik tamamlandı!"
echo ""
echo "📝 Şimdi yapılacaklar:"
echo "  1. Yeni test çalıştır: python pipeline_runner.py"
echo "  2. Sonuçları HITL'ye yükle"
