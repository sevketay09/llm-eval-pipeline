#!/bin/bash
# =============================================================================
# LLM Evaluation Pipeline - Quick Start Script
# =============================================================================

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

echo "==================================================================="
echo "LLM Evaluation Pipeline - Available Commands"
echo "==================================================================="
echo ""
echo "1. Test Cloud Models (Azure GPT-4o):"
echo "   python main.py --models gpt4o-azure --suite smoke"
echo ""
echo "2. Test On-Prem Models (Qwen & Mistral):"
echo "   python main.py --models qwen-31-onprem mistral-small-31-onprem --suite smoke"
echo ""
echo "3. Compare Cloud vs On-Prem:"
echo "   python main.py --models gpt4o-azure qwen-31-onprem --suite full"
echo ""
echo "4. Full Fintech Benchmark:"
echo "   python main.py --models gpt4o-azure qwen-31-onprem --suite fintech_only"
echo ""
echo "5. Run Dashboard (Streamlit):"
echo "   streamlit run app.py"
echo ""
echo "==================================================================="
echo ""

# Check current environment
echo "Current Environment:"
echo "  LLM_API_URL: ${LLM_API_URL:-NOT SET}"
echo "  MISTRAL_API_URL: ${MISTRAL_API_URL:-NOT SET}"
echo "  AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT:-NOT SET}"
echo ""
