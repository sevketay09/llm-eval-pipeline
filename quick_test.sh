#!/bin/bash
# Quick Smoke Test - 5 dakikada model testi
# Usage: ./quick_test.sh [model-name]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL="${1:-qwen-31-onprem}"

echo -e "${BLUE}🚀 Quick Smoke Test${NC}"
echo -e "   Model: ${YELLOW}$MODEL${NC}"
echo -e ""

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run smoke test (only 5 samples per test)
python main.py --models "$MODEL" --suite smoke

echo -e ""
echo -e "${GREEN}✅ Smoke test completed!${NC}"
echo -e "${BLUE}💡 For full test: ${YELLOW}./run.sh $MODEL full${NC}"
