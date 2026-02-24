#!/bin/bash

# LLM Evaluation Pipeline - Dashboard Başlatıcı
# ===============================================

set -e

echo "🚀 LLM Evaluation Pipeline Dashboard başlatılıyor..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment bulunamadı!"
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📥 Bağımlılıklar yükleniyor..."
    pip install -r requirements.txt
else
    echo "✅ Virtual environment bulundu"
    source venv/bin/activate
fi

echo ""
echo "🌐 Dashboard açılıyor..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "⏹️  Durdurmak için CTRL+C"
echo ""

# Start streamlit
streamlit run app.py
