"""
Latency Test Script

지연시간 측정 벤치마크

실행 방법:
    python benchmarks/latency_test.py --num-requests 100
"""

import argparse
import asyncio
import csv
import os
import statistics
import time
from datetime import datetime
from typing import List, Dict, Any

import httpx


# ===========================================
# Configuration
# ===========================================
DEFAULT_URL = "http://localhost:8000/api/v1/chat/completions"
DEFAULT_NUM_REQUESTS = 100
DEFAULT_MAX_TOKENS = 256
WARMUP_REQUESTS = 5

TEST_MESSAGES = [
    "내일 오전 10시 기온을 예측해주세요.",
    "토마토 100평 재배 시 예상 수확량을 알려주세요.",
    "태양광 패널 최적 각도를 계산해주세요.",
    "서울 다음 주 날씨를 알려주세요.",
]


# ===========================================
# Latency Measurement
# ===========================================
async def measure_single_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    request_num: int
) -> Dict[str, Any]:
    """단일 요청 지연시간 측정"""
    start_time = time.time()

    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        data = response.json()

        return {
            "request_num": request_num,
            "latency_ms": latency_ms,
            "tokens_used": data.get("tokens_used", 0),
            "success": True,
            "error": None
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "request_num": request_num,
            "latency_ms": latency_ms,
            "tokens_used": 0,
            "success": False,
            "error": str(e)
        }


async def run_latency_test(
    url: str,
    num_requests: int,
    max_tokens: int,
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    """지연시간 테스트 실행"""

    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warmup
        print(f"Warming up with {WARMUP_REQUESTS} requests...")
        for i in range(WARMUP_REQUESTS):
            payload = {
                "message": TEST_MESSAGES[i % len(TEST_MESSAGES)],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            await measure_single_request(client, url, payload, -1)

        print(f"\nRunning {num_requests} requests...")
        print("-" * 60)

        for i in range(num_requests):
            payload = {
                "message": TEST_MESSAGES[i % len(TEST_MESSAGES)],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            result = await measure_single_request(client, url, payload, i + 1)
            results.append(result)

            # Progress 출력
            status = "OK" if result["success"] else "FAIL"
            print(f"Request {i + 1:3d}/{num_requests}: {result['latency_ms']:7.2f}ms [{status}]")

    return results


# ===========================================
# Statistics Calculation
# ===========================================
def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """통계 계산"""
    successful = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful]
    tokens = [r["tokens_used"] for r in successful]

    if not latencies:
        return {"error": "No successful requests"}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    stats = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "success_rate": len(successful) / len(results) * 100,

        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "latency_p50_ms": sorted_latencies[int(n * 0.50)],
        "latency_p75_ms": sorted_latencies[int(n * 0.75)],
        "latency_p90_ms": sorted_latencies[int(n * 0.90)],
        "latency_p95_ms": sorted_latencies[int(n * 0.95)],
        "latency_p99_ms": sorted_latencies[min(int(n * 0.99), n - 1)],

        "total_tokens": sum(tokens),
        "avg_tokens_per_request": statistics.mean(tokens) if tokens else 0,
    }

    return stats


# ===========================================
# Output Functions
# ===========================================
def print_statistics(stats: Dict[str, Any]):
    """통계 출력"""
    print("\n" + "=" * 60)
    print("LATENCY TEST RESULTS")
    print("=" * 60)

    print(f"\n📊 Summary:")
    print(f"   Total Requests:      {stats['total_requests']}")
    print(f"   Successful:          {stats['successful_requests']}")
    print(f"   Failed:              {stats['failed_requests']}")
    print(f"   Success Rate:        {stats['success_rate']:.2f}%")

    print(f"\n⏱️ Latency (ms):")
    print(f"   Mean:                {stats['latency_mean_ms']:.2f}")
    print(f"   Median (P50):        {stats['latency_median_ms']:.2f}")
    print(f"   Std Dev:             {stats['latency_stdev_ms']:.2f}")
    print(f"   Min:                 {stats['latency_min_ms']:.2f}")
    print(f"   Max:                 {stats['latency_max_ms']:.2f}")

    print(f"\n📈 Percentiles (ms):")
    print(f"   P50:                 {stats['latency_p50_ms']:.2f}")
    print(f"   P75:                 {stats['latency_p75_ms']:.2f}")
    print(f"   P90:                 {stats['latency_p90_ms']:.2f}")
    print(f"   P95:                 {stats['latency_p95_ms']:.2f}")
    print(f"   P99:                 {stats['latency_p99_ms']:.2f}")

    print(f"\n🔤 Tokens:")
    print(f"   Total:               {stats['total_tokens']}")
    print(f"   Avg per Request:     {stats['avg_tokens_per_request']:.1f}")

    print("\n" + "=" * 60)


def save_results(results: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_path = os.path.join(output_dir, f"latency_results_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_num", "latency_ms", "tokens_used", "success", "error"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")

    # Summary JSON 저장
    import json
    summary_path = os.path.join(output_dir, f"latency_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary saved to: {summary_path}")


# ===========================================
# Main
# ===========================================
async def main():
    parser = argparse.ArgumentParser(description="Latency Test for vLLM API")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API URL")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS, help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Latency Test")
    print("=" * 60)
    print(f"URL:          {args.url}")
    print(f"Requests:     {args.num_requests}")
    print(f"Max Tokens:   {args.max_tokens}")
    print(f"Temperature:  {args.temperature}")
    print("=" * 60)

    # 테스트 실행
    results = await run_latency_test(
        url=args.url,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # 통계 계산 및 출력
    stats = calculate_statistics(results)
    print_statistics(stats)

    # 결과 저장
    if not args.no_save:
        save_results(results, stats, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
