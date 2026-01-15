#!/bin/bash
# ===========================================
# Stop Services Script
# ===========================================

echo "=========================================="
echo "Stopping vLLM Serving Services"
echo "=========================================="

# 서비스 중지
echo "Stopping containers..."
docker-compose down

echo ""
echo "Services stopped."

# 정리 옵션
read -p "Remove model cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing model cache..."
    rm -rf ./models_cache
    echo "Model cache removed."
fi

read -p "Remove benchmark results? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing benchmark results..."
    rm -f ./benchmarks/results/*.csv
    rm -f ./benchmarks/results/*.json
    echo "Benchmark results removed."
fi

echo ""
echo "Done!"
