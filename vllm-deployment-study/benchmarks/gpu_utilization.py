"""
GPU Utilization Monitor

GPU 사용률 모니터링 스크립트

실행 방법:
    python benchmarks/gpu_utilization.py 60  # 60초 동안 모니터링
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional


# ===========================================
# GPU Monitoring
# ===========================================
def get_gpu_info() -> Optional[Dict[str, float]]:
    """nvidia-smi를 통해 GPU 정보 조회"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        output = result.stdout.strip()
        if not output:
            return None

        # 첫 번째 GPU만 파싱 (멀티 GPU 확장 가능)
        values = output.split("\n")[0].split(", ")

        return {
            "gpu_index": int(values[0]),
            "gpu_name": values[1].strip(),
            "gpu_utilization": float(values[2]),
            "memory_used_mb": float(values[3]),
            "memory_total_mb": float(values[4]),
            "memory_utilization": float(values[3]) / float(values[4]) * 100,
            "temperature_c": float(values[5]),
            "power_draw_w": float(values[6]) if values[6] != "[N/A]" else 0
        }

    except FileNotFoundError:
        print("Error: nvidia-smi not found. Is NVIDIA driver installed?")
        return None
    except subprocess.TimeoutExpired:
        print("Warning: nvidia-smi timeout")
        return None
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None


def monitor_gpu(
    duration_seconds: int,
    interval: float = 1.0,
    output_file: Optional[str] = None
) -> List[Dict]:
    """GPU 사용률 모니터링"""

    results = []
    start_time = time.time()

    print("=" * 80)
    print("GPU Utilization Monitor")
    print("=" * 80)

    # 초기 GPU 정보
    initial_info = get_gpu_info()
    if initial_info:
        print(f"GPU: {initial_info['gpu_name']}")
        print(f"Total Memory: {initial_info['memory_total_mb']:.0f} MB")
    else:
        print("Warning: Could not get GPU info")
        print("Make sure NVIDIA driver is installed and nvidia-smi is available")
        return []

    print("-" * 80)
    print(f"{'Time':>10} | {'GPU %':>7} | {'Mem Used':>10} | {'Mem %':>7} | {'Temp':>6} | {'Power':>8}")
    print("-" * 80)

    try:
        while time.time() - start_time < duration_seconds:
            info = get_gpu_info()

            if info:
                timestamp = datetime.now().strftime("%H:%M:%S")
                elapsed = time.time() - start_time

                # 결과 저장
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 2),
                    **info
                }
                results.append(record)

                # 출력
                print(f"{timestamp:>10} | "
                      f"{info['gpu_utilization']:>6.1f}% | "
                      f"{info['memory_used_mb']:>8.0f}MB | "
                      f"{info['memory_utilization']:>6.1f}% | "
                      f"{info['temperature_c']:>4.0f}°C | "
                      f"{info['power_draw_w']:>6.1f}W")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")

    return results


# ===========================================
# Statistics
# ===========================================
def calculate_statistics(results: List[Dict]) -> Dict:
    """통계 계산"""
    if not results:
        return {}

    gpu_utils = [r["gpu_utilization"] for r in results]
    mem_utils = [r["memory_utilization"] for r in results]
    temps = [r["temperature_c"] for r in results]
    powers = [r["power_draw_w"] for r in results if r["power_draw_w"] > 0]

    stats = {
        "samples": len(results),
        "duration_seconds": results[-1]["elapsed_seconds"],

        "gpu_util_avg": sum(gpu_utils) / len(gpu_utils),
        "gpu_util_min": min(gpu_utils),
        "gpu_util_max": max(gpu_utils),

        "mem_util_avg": sum(mem_utils) / len(mem_utils),
        "mem_util_min": min(mem_utils),
        "mem_util_max": max(mem_utils),

        "memory_used_avg_mb": sum(r["memory_used_mb"] for r in results) / len(results),
        "memory_total_mb": results[0]["memory_total_mb"],

        "temp_avg_c": sum(temps) / len(temps),
        "temp_max_c": max(temps),
    }

    if powers:
        stats["power_avg_w"] = sum(powers) / len(powers)
        stats["power_max_w"] = max(powers)

    return stats


def print_statistics(stats: Dict):
    """통계 출력"""
    if not stats:
        print("No statistics available")
        return

    print("\n" + "=" * 60)
    print("GPU UTILIZATION STATISTICS")
    print("=" * 60)

    print(f"\n📊 Monitoring Summary:")
    print(f"   Samples:             {stats['samples']}")
    print(f"   Duration:            {stats['duration_seconds']:.1f}s")

    print(f"\n🖥️ GPU Utilization:")
    print(f"   Average:             {stats['gpu_util_avg']:.1f}%")
    print(f"   Min:                 {stats['gpu_util_min']:.1f}%")
    print(f"   Max:                 {stats['gpu_util_max']:.1f}%")

    print(f"\n💾 Memory Utilization:")
    print(f"   Average:             {stats['mem_util_avg']:.1f}%")
    print(f"   Average Used:        {stats['memory_used_avg_mb']:.0f} MB / {stats['memory_total_mb']:.0f} MB")
    print(f"   Min:                 {stats['mem_util_min']:.1f}%")
    print(f"   Max:                 {stats['mem_util_max']:.1f}%")

    print(f"\n🌡️ Temperature:")
    print(f"   Average:             {stats['temp_avg_c']:.1f}°C")
    print(f"   Max:                 {stats['temp_max_c']:.1f}°C")

    if "power_avg_w" in stats:
        print(f"\n⚡ Power:")
        print(f"   Average:             {stats['power_avg_w']:.1f}W")
        print(f"   Max:                 {stats['power_max_w']:.1f}W")

    print("\n" + "=" * 60)


# ===========================================
# Save Results
# ===========================================
def save_results(results: List[Dict], stats: Dict, output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    if results:
        csv_path = os.path.join(output_dir, f"gpu_utilization_{timestamp}.csv")
        fieldnames = ["timestamp", "elapsed_seconds", "gpu_utilization",
                      "memory_used_mb", "memory_utilization", "temperature_c", "power_draw_w"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"Data saved to: {csv_path}")

    # 통계 JSON 저장
    if stats:
        import json
        json_path = os.path.join(output_dir, f"gpu_stats_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {json_path}")


# ===========================================
# Main
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="GPU Utilization Monitor")
    parser.add_argument("duration", type=int, nargs="?", default=60,
                        help="Monitoring duration in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results",
                        help="Output directory")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results")

    args = parser.parse_args()

    # 모니터링 실행
    results = monitor_gpu(
        duration_seconds=args.duration,
        interval=args.interval
    )

    if results:
        # 통계 계산 및 출력
        stats = calculate_statistics(results)
        print_statistics(stats)

        # 결과 저장
        if not args.no_save:
            save_results(results, stats, args.output_dir)
    else:
        print("\nNo data collected. Check if nvidia-smi is available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
