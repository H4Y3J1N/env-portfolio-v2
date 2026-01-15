"""
Visualization Utilities

청킹 및 검색 결과 시각화 함수들
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트 설정 (시스템에 따라 다름)
try:
    # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        pass

plt.rcParams['axes.unicode_minus'] = False


def plot_chunk_distribution(
    chunk_sizes: Dict[str, List[int]],
    save_path: Optional[str] = None,
    title: str = "Chunk Size Distribution by Strategy"
) -> plt.Figure:
    """
    청크 크기 분포 시각화

    Args:
        chunk_sizes: {strategy: [sizes], ...}
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, len(chunk_sizes), figsize=(5 * len(chunk_sizes), 4))

    if len(chunk_sizes) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (strategy, sizes) in enumerate(chunk_sizes.items()):
        ax = axes[idx]

        ax.hist(sizes, bins=30, color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')
        ax.axvline(np.median(sizes), color='green', linestyle=':', label=f'Median: {np.median(sizes):.0f}')

        ax.set_xlabel('Chunk Size (words)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{strategy.capitalize()}\n(n={len(sizes)}, std={np.std(sizes):.1f})')
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Retrieval Metrics Comparison"
) -> plt.Figure:
    """
    메트릭 비교 시각화

    Args:
        metrics: {strategy: {metric: value, ...}, ...}
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure
    """
    strategies = list(metrics.keys())
    metric_names = ['precision@5', 'recall@5', 'f1@5', 'mrr']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 2, idx % 2]

        values = []
        for strategy in strategies:
            val = metrics[strategy].get(metric, 0)
            values.append(val)

        bars = ax.bar(strategies, values, color=colors[:len(strategies)])

        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylim(0, 1)

        # 값 표시
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_latency_comparison(
    latencies: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Processing Latency Comparison"
) -> plt.Figure:
    """
    처리 시간 비교 시각화

    Args:
        latencies: {strategy: latency_seconds, ...}
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = list(latencies.keys())
    values = list(latencies.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(strategies, values, color=colors[:len(strategies)])

    ax.set_ylabel('Latency (seconds)')
    ax.set_title(title)

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{val:.2f}s',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_difficulty_analysis(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Performance by Query Difficulty"
) -> plt.Figure:
    """
    난이도별 성능 분석 시각화

    Args:
        results: {strategy: {difficulty: f1_score, ...}, ...}
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = list(results.keys())
    difficulties = ['easy', 'medium', 'hard']
    x = np.arange(len(difficulties))
    width = 0.25

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, strategy in enumerate(strategies):
        values = [results[strategy].get(diff, 0) for diff in difficulties]
        bars = ax.bar(x + i * width, values, width, label=strategy.capitalize(), color=colors[i])

        # 값 표시
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    ax.set_xlabel('Difficulty')
    ax.set_ylabel('F1@5 Score')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_confusion_matrix(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    검색 결과 히트맵 (쿼리 x 청크)

    Args:
        results: 평가 결과
        save_path: 저장 경로

    Returns:
        matplotlib Figure
    """
    # 간단한 히트맵 예시
    fig, ax = plt.subplots(figsize=(10, 8))

    # 결과에서 데이터 추출
    detailed = results.get("detailed_results", [])

    if not detailed:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig

    # 쿼리별 성능 매트릭스
    query_ids = [r["query_id"] for r in detailed[:20]]  # 최대 20개
    metrics = ["precision@5", "recall@5", "f1@5", "mrr"]

    data = []
    for r in detailed[:20]:
        row = [
            r.get("precision", {}).get(5, 0),
            r.get("recall", {}).get(5, 0),
            r.get("f1", {}).get(5, 0),
            r.get("mrr", 0),
        ]
        data.append(row)

    data = np.array(data)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(query_ids)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(query_ids)

    plt.colorbar(im, ax=ax, label='Score')

    ax.set_title('Query-wise Performance Heatmap')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Query ID')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def create_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> str:
    """
    비교 표 생성 (마크다운)

    Args:
        all_results: {strategy: evaluation_results, ...}
        save_path: 저장 경로

    Returns:
        마크다운 테이블 문자열
    """
    headers = ["Strategy", "Precision@5", "Recall@5", "F1@5", "MRR", "MAP"]

    rows = []
    for strategy, results in all_results.items():
        metrics = results.get("metrics", {})
        row = [
            strategy.capitalize(),
            f"{metrics.get('precision@5', 0):.3f}",
            f"{metrics.get('recall@5', 0):.3f}",
            f"{metrics.get('f1@5', 0):.3f}",
            f"{metrics.get('mrr', 0):.3f}",
            f"{metrics.get('map', 0):.3f}",
        ]
        rows.append(row)

    # 마크다운 테이블 생성
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"Saved table to {save_path}")

    return table
