#!/bin/bash
# LLM Evaluation Pipeline - Quick Run Script
# Tek komut ile modelleri test et

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        LLM Evaluation Pipeline                          ║${NC}"
echo -e "${BLUE}║        On-Prem & Cloud Model Testing                    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo -e "${YELLOW}📝 Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file with your API keys and URLs${NC}"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Default values
MODELS="${1:-qwen-31-onprem,mistral-small-31-onprem}"
SUITE="${2:-full}"
REPORT_DIR="reports"

# Print configuration
echo -e "${GREEN}📋 Configuration:${NC}"
echo -e "   Models: ${YELLOW}$MODELS${NC}"
echo -e "   Test Suite: ${YELLOW}$SUITE${NC}"
echo -e ""

# Check Python dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"
if ! python -c "import openai, anthropic, yaml, tqdm" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    pip install -r requirements.txt
fi

# Create reports directory
mkdir -p "$REPORT_DIR"

# Run evaluation
echo -e "${GREEN}🚀 Starting evaluation...${NC}"
echo -e ""

# Convert comma-separated models to space-separated
MODELS_ARRAY=(${MODELS//,/ })

python main.py \
    --models "${MODELS_ARRAY[@]}" \
    --suite "$SUITE" \
    --config config/models.yaml

# Check exit code
if [ $? -eq 0 ]; then
    echo -e ""
    echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
    echo -e ""
    echo -e "${BLUE}📊 View results:${NC}"
    echo -e "   Dashboard: ${YELLOW}streamlit run dashboard.py${NC}"
    echo -e "   Reports: ${YELLOW}ls -lth $REPORT_DIR/${NC}"
    echo -e ""
else
    echo -e ""
    echo -e "${RED}❌ Evaluation failed! Check logs above.${NC}"
    exit 1
fi

# Quick summary
echo -e "${BLUE}📈 Quick Summary:${NC}"
LATEST_REPORT=$(ls -t $REPORT_DIR/eval_*.json 2>/dev/null | head -1)
if [ -f "$LATEST_REPORT" ]; then
    echo -e "   Latest report: ${YELLOW}$LATEST_REPORT${NC}"
    # Extract summary if possible (basic JSON parsing)
    if command -v jq &> /dev/null; then
        echo -e ""
        jq -r '.summary.model_comparison | to_entries[] | "   \(.key): Overall Score = \(.value.overall_score)"' "$LATEST_REPORT" 2>/dev/null || true
    fi
fi

echo -e ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 Evaluation Complete! 🎉                  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
