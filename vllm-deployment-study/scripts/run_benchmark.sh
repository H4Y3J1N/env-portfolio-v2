#!/bin/bash
# ===========================================
# Run Benchmark Script
# ===========================================

set -e

echo "=========================================="
echo "vLLM Benchmark Suite"
echo "=========================================="

# 결과 디렉토리 생성
mkdir -p benchmarks/results

# Health check
echo ""
echo "Checking service health..."
if ! curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "Error: API server is not healthy."
    echo "Please start services first: ./scripts/start_services.sh"
    exit 1
fi
echo "Services are healthy!"

# 벤치마크 메뉴
echo ""
echo "Select benchmark to run:"
echo "  1) Latency Test (single request latency)"
echo "  2) Throughput Test (concurrent requests)"
echo "  3) GPU Utilization Monitor"
echo "  4) Full Benchmark Suite"
echo "  5) Load Test (Locust - interactive)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Running Latency Test..."
        python benchmarks/latency_test.py --num-requests 50 --max-tokens 128
        ;;
    2)
        echo ""
        echo "Running Throughput Test..."
        python benchmarks/throughput_test.py --multi-level --duration 30
        ;;
    3)
        echo ""
        echo "Running GPU Utilization Monitor (60 seconds)..."
        python benchmarks/gpu_utilization.py 60
        ;;
    4)
        echo ""
        echo "Running Full Benchmark Suite..."
        echo ""

        echo "Step 1/3: Latency Test"
        echo "------------------------"
        python benchmarks/latency_test.py --num-requests 100 --max-tokens 128

        echo ""
        echo "Step 2/3: Throughput Test"
        echo "------------------------"
        python benchmarks/throughput_test.py --multi-level --duration 30

        echo ""
        echo "Step 3/3: GPU Monitoring (30 seconds)"
        echo "------------------------"
        python benchmarks/gpu_utilization.py 30

        echo ""
        echo "=========================================="
        echo "Full Benchmark Complete!"
        echo "Results saved to: benchmarks/results/"
        echo "=========================================="
        ;;
    5)
        echo ""
        echo "Starting Locust Load Test..."
        echo "Web UI will be available at: http://localhost:8089"
        echo ""
        locust -f benchmarks/load_test.py --host http://localhost:8000
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"
