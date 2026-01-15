"""
Throughput Test Script

처리량(Requests/sec, Tokens/sec) 측정 벤치마크

실행 방법:
    python benchmarks/throughput_test.py --concurrent-users 10 --duration 60
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

import httpx


# ===========================================
# Configuration
# ===========================================
DEFAULT_URL = "http://localhost:8000/api/v1/chat/completions"
DEFAULT_CONCURRENT_USERS = 10
DEFAULT_DURATION_SECONDS = 60
DEFAULT_MAX_TOKENS = 128

TEST_MESSAGES = [
    "내일 기온을 예측해주세요.",
    "토마토 수확량 예측해줘.",
    "패널 각도 최적화해줘.",
    "오늘 날씨 알려줘.",
]


# ===========================================
# Worker Function
# ===========================================
async def worker(
    worker_id: int,
    url: str,
    payload: dict,
    results: Dict[str, Any],
    stop_event: asyncio.Event
):
    """개별 워커 - 지속적으로 요청 전송"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        request_num = 0

        while not stop_event.is_set():
            start_time = time.time()
            request_num += 1

            # 메시지 순환
            current_payload = payload.copy()
            current_payload["message"] = TEST_MESSAGES[request_num % len(TEST_MESSAGES)]

            try:
                response = await client.post(url, json=current_payload)
                response.raise_for_status()

                latency_ms = (time.time() - start_time) * 1000
                data = response.json()
                tokens = data.get("tokens_used", 0)

                results["total_requests"] += 1
                results["total_tokens"] += tokens
                results["latencies"].append(latency_ms)

            except asyncio.CancelledError:
                break
            except Exception as e:
                results["errors"] += 1
                # 에러 시 잠시 대기
                await asyncio.sleep(0.5)


# ===========================================
# Throughput Measurement
# ===========================================
async def run_throughput_test(
    url: str,
    concurrent_users: int,
    duration_seconds: int,
    max_tokens: int,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """처리량 테스트 실행"""

    print(f"\nStarting throughput test:")
    print(f"  - Concurrent Users: {concurrent_users}")
    print(f"  - Duration: {duration_seconds}s")
    print(f"  - Max Tokens: {max_tokens}")
    print("-" * 60)

    # 공유 결과 저장소
    results = defaultdict(int)
    results["latencies"] = []

    # 기본 페이로드
    payload = {
        "message": "",
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # 중지 이벤트
    stop_event = asyncio.Event()

    # 워커 시작
    start_time = time.time()
    tasks = [
        asyncio.create_task(
            worker(i, url, payload, results, stop_event)
        )
        for i in range(concurrent_users)
    ]

    # 진행 상황 출력
    print("\nProgress:")
    elapsed = 0
    while elapsed < duration_seconds:
        await asyncio.sleep(5)
        elapsed = time.time() - start_time

        current_rps = results["total_requests"] / elapsed if elapsed > 0 else 0
        current_tps = results["total_tokens"] / elapsed if elapsed > 0 else 0

        print(f"  [{elapsed:.0f}s] Requests: {results['total_requests']}, "
              f"RPS: {current_rps:.2f}, TPS: {current_tps:.2f}")

        if elapsed >= duration_seconds:
            break

    # 워커 중지
    stop_event.set()

    # 태스크 취소 및 정리
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 계산
    total_time = time.time() - start_time

    return {
        "duration_seconds": total_time,
        "concurrent_users": concurrent_users,
        "total_requests": results["total_requests"],
        "total_tokens": results["total_tokens"],
        "errors": results["errors"],
        "requests_per_second": results["total_requests"] / total_time,
        "tokens_per_second": results["total_tokens"] / total_time,
        "success_rate": (results["total_requests"] / (results["total_requests"] + results["errors"])) * 100
        if (results["total_requests"] + results["errors"]) > 0 else 0,
        "latencies": results["latencies"]
    }


# ===========================================
# Multi-Level Test
# ===========================================
async def run_multi_level_test(
    url: str,
    user_levels: List[int],
    duration_per_level: int,
    max_tokens: int
) -> List[Dict[str, Any]]:
    """여러 동시 사용자 수준으로 테스트"""

    all_results = []

    for concurrent_users in user_levels:
        print(f"\n{'=' * 60}")
        print(f"Testing with {concurrent_users} concurrent users")
        print("=" * 60)

        result = await run_throughput_test(
            url=url,
            concurrent_users=concurrent_users,
            duration_seconds=duration_per_level,
            max_tokens=max_tokens
        )

        # 레이턴시 통계 추가
        if result["latencies"]:
            latencies = sorted(result["latencies"])
            n = len(latencies)
            result["latency_mean_ms"] = sum(latencies) / n
            result["latency_p50_ms"] = latencies[int(n * 0.50)]
            result["latency_p95_ms"] = latencies[int(n * 0.95)]
            result["latency_p99_ms"] = latencies[min(int(n * 0.99), n - 1)]
        else:
            result["latency_mean_ms"] = 0
            result["latency_p50_ms"] = 0
            result["latency_p95_ms"] = 0
            result["latency_p99_ms"] = 0

        # 레이턴시 배열 제거 (저장 용량 절약)
        del result["latencies"]

        all_results.append(result)

        # 쿨다운
        print("\nCooling down for 5 seconds...")
        await asyncio.sleep(5)

    return all_results


# ===========================================
# Output Functions
# ===========================================
def print_results(results: List[Dict[str, Any]]):
    """결과 출력"""
    print("\n" + "=" * 80)
    print("THROUGHPUT TEST RESULTS")
    print("=" * 80)

    # 테이블 헤더
    print(f"\n{'Users':>6} | {'Requests':>10} | {'RPS':>10} | {'TPS':>12} | "
          f"{'P50(ms)':>10} | {'P95(ms)':>10} | {'Success':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['concurrent_users']:>6} | "
              f"{r['total_requests']:>10} | "
              f"{r['requests_per_second']:>10.2f} | "
              f"{r['tokens_per_second']:>12.2f} | "
              f"{r['latency_p50_ms']:>10.2f} | "
              f"{r['latency_p95_ms']:>10.2f} | "
              f"{r['success_rate']:>7.1f}%")

    print("-" * 80)

    # Best RPS 찾기
    best = max(results, key=lambda x: x["requests_per_second"])
    print(f"\n🏆 Best Throughput: {best['requests_per_second']:.2f} req/s "
          f"@ {best['concurrent_users']} concurrent users")


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON 저장
    json_path = os.path.join(output_dir, f"throughput_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # CSV 저장
    csv_path = os.path.join(output_dir, f"throughput_results_{timestamp}.csv")
    with open(csv_path, "w") as f:
        headers = ["concurrent_users", "total_requests", "requests_per_second",
                   "tokens_per_second", "latency_p50_ms", "latency_p95_ms", "success_rate"]
        f.write(",".join(headers) + "\n")
        for r in results:
            values = [str(r.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")
    print(f"CSV saved to: {csv_path}")


# ===========================================
# Main
# ===========================================
async def main():
    parser = argparse.ArgumentParser(description="Throughput Test for vLLM API")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API URL")
    parser.add_argument("--concurrent-users", type=int, default=DEFAULT_CONCURRENT_USERS,
                        help="Number of concurrent users (single level test)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_SECONDS,
                        help="Test duration in seconds")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Max tokens per request")
    parser.add_argument("--multi-level", action="store_true",
                        help="Run multi-level test (1, 5, 10, 20 users)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results",
                        help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Throughput Test")
    print("=" * 60)
    print(f"URL:          {args.url}")
    print(f"Max Tokens:   {args.max_tokens}")
    print(f"Duration:     {args.duration}s per level")

    if args.multi_level:
        # 다중 레벨 테스트
        user_levels = [1, 5, 10, 20]
        print(f"User Levels:  {user_levels}")
        results = await run_multi_level_test(
            url=args.url,
            user_levels=user_levels,
            duration_per_level=args.duration,
            max_tokens=args.max_tokens
        )
    else:
        # 단일 레벨 테스트
        print(f"Users:        {args.concurrent_users}")
        result = await run_throughput_test(
            url=args.url,
            concurrent_users=args.concurrent_users,
            duration_seconds=args.duration,
            max_tokens=args.max_tokens
        )

        # 레이턴시 통계 추가
        if result["latencies"]:
            latencies = sorted(result["latencies"])
            n = len(latencies)
            result["latency_mean_ms"] = sum(latencies) / n
            result["latency_p50_ms"] = latencies[int(n * 0.50)]
            result["latency_p95_ms"] = latencies[int(n * 0.95)]
            result["latency_p99_ms"] = latencies[min(int(n * 0.99), n - 1)]
        del result["latencies"]

        results = [result]

    # 결과 출력
    print_results(results)

    # 결과 저장
    if not args.no_save:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
