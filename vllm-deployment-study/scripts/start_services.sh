#!/bin/bash
# ===========================================
# Start Services Script
# ===========================================

set -e

echo "=========================================="
echo "Starting vLLM Serving Services"
echo "=========================================="

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 환경 변수 확인
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file and set HF_TOKEN${NC}"
fi

# GPU 확인
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}GPU detected!${NC}"
else
    echo -e "${RED}Warning: nvidia-smi not found. GPU may not be available.${NC}"
fi

# 이전 컨테이너 정리
echo ""
echo "Cleaning up previous containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# 서비스 시작
echo ""
echo "Starting services..."
docker-compose up -d

# 상태 확인
echo ""
echo "Waiting for services to be ready..."
echo "(Inference server may take 2-3 minutes to load the model)"

# Health check 대기
MAX_RETRIES=60
RETRY_INTERVAL=5

for i in $(seq 1 $MAX_RETRIES); do
    echo -n "."

    # Inference server health check
    if curl -s http://localhost:8001/health | grep -q "healthy"; then
        echo ""
        echo -e "${GREEN}Inference server is ready!${NC}"
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        echo ""
        echo -e "${YELLOW}Inference server is still starting...${NC}"
        echo "Check logs with: docker-compose logs -f inference-server"
    fi

    sleep $RETRY_INTERVAL
done

# API server health check
echo ""
echo "Checking API server..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}API server is ready!${NC}"
else
    echo -e "${YELLOW}API server may still be starting...${NC}"
fi

# 최종 상태
echo ""
echo "=========================================="
echo "Service Status:"
echo "=========================================="
docker-compose ps

echo ""
echo "=========================================="
echo "Endpoints:"
echo "=========================================="
echo "  API Server:       http://localhost:8000"
echo "  API Docs:         http://localhost:8000/docs"
echo "  Inference Server: http://localhost:8001"
echo "  Health Check:     http://localhost:8000/health"

echo ""
echo "=========================================="
echo "Quick Test:"
echo "=========================================="
echo 'curl -X POST http://localhost:8000/api/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"message": "안녕하세요!", "max_tokens": 100}'"'"

echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: ./scripts/stop_services.sh"
