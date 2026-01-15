"""
시각화 모듈

이 모듈은 실험 결과의 시각화 기능을 제공합니다.
"""

import os
import json
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정 시도
def setup_korean_font():
    """한글 폰트 설정"""
    # Windows
    font_candidates = [
        'Malgun Gothic',  # Windows
        'AppleGothic',    # macOS
        'NanumGothic',    # Linux
        'DejaVu Sans',    # Fallback
    ]

    for font_name in font_candidates:
        try:
            fm.findfont(font_name)
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            continue

    print("Warning: Korean font not found. Using default font.")

setup_korean_font()

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def plot_loss_curve(
    log_history: List[Dict],
    output_path: str,
    title: str = "Training Loss Curve"
) -> None:
    """학습 손실 곡선 플롯

    Args:
        log_history: 학습 로그 히스토리
        output_path: 저장 경로
        title: 그래프 제목
    """
    # 데이터 추출
    train_loss = [(x['step'], x['loss']) for x in log_history if 'loss' in x and 'eval_loss' not in x]
    eval_loss = [(x['step'], x['eval_loss']) for x in log_history if 'eval_loss' in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training Loss
    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0].plot(steps, losses, color=COLORS[0], linewidth=2, label='Train Loss')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Validation Loss
    if eval_loss:
        steps, losses = zip(*eval_loss)
        axes[1].plot(steps, losses, color=COLORS[1], linewidth=2, label='Eval Loss')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to: {output_path}")


def plot_tool_accuracy(
    tool_accuracy: Dict[str, float],
    tool_stats: Dict[str, Dict],
    output_path: str,
    title: str = "Tool Calling Accuracy by Tool"
) -> None:
    """Tool별 정확도 바 차트

    Args:
        tool_accuracy: Tool별 정확도 딕셔너리
        tool_stats: Tool별 통계
        output_path: 저장 경로
        title: 그래프 제목
    """
    tools = list(tool_accuracy.keys())
    accuracies = [tool_accuracy[t] for t in tools]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(tools, accuracies, color=COLORS[:len(tools)])

    ax.set_xlabel('Accuracy')
    ax.set_title(title)
    ax.set_xlim(0, 1)

    # 값 표시
    for i, (bar, tool) in enumerate(zip(bars, tools)):
        width = bar.get_width()
        stats = tool_stats.get(tool, {})
        correct = stats.get('correct', 0)
        total = stats.get('total', 0)
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.1%} ({correct}/{total})',
                va='center', fontsize=10)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Tool accuracy chart saved to: {output_path}")


def plot_error_distribution(
    error_distribution: Dict[str, int],
    output_path: str,
    title: str = "Error Type Distribution"
) -> None:
    """오류 타입 분포 파이 차트

    Args:
        error_distribution: 오류 타입별 카운트
        output_path: 저장 경로
        title: 그래프 제목
    """
    # 0이 아닌 오류만 필터링
    errors = {k: v for k, v in error_distribution.items() if v > 0}

    if not errors:
        print("No errors to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(errors.keys())
    sizes = list(errors.values())

    # 레이블 정리
    label_map = {
        "no_tool_call": "Tool 호출 안함",
        "unexpected_tool_call": "불필요한 Tool 호출",
        "wrong_tool": "잘못된 Tool",
        "missing_argument": "인자 누락",
        "wrong_argument_value": "잘못된 인자 값",
    }
    labels = [label_map.get(l, l) for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=COLORS[:len(sizes)],
        startangle=90
    )

    ax.set_title(title)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Error distribution chart saved to: {output_path}")


def create_comparison_chart(
    experiments: List[Dict[str, Any]],
    output_path: str,
    title: str = "Experiment Comparison"
) -> None:
    """여러 실험 결과 비교 차트

    Args:
        experiments: 실험 결과 리스트
            [{"name": "exp-001", "accuracy": 0.65, "config": {...}}, ...]
        output_path: 저장 경로
        title: 그래프 제목
    """
    df = pd.DataFrame(experiments)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 전체 정확도 비교
    ax1 = axes[0, 0]
    bars = ax1.bar(df['name'], df['accuracy'], color=COLORS[:len(df)])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Tool Calling Accuracy')
    ax1.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

    # 2. Learning Rate vs Accuracy
    ax2 = axes[0, 1]
    if 'learning_rate' in df.columns:
        ax2.scatter(df['learning_rate'], df['accuracy'], s=100, color=COLORS[0])
        for i, row in df.iterrows():
            ax2.annotate(row['name'], (row['learning_rate'], row['accuracy']),
                        textcoords="offset points", xytext=(5, 5))
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Learning Rate vs Accuracy')
        ax2.set_xscale('log')

    # 3. LoRA Rank vs Accuracy
    ax3 = axes[1, 0]
    if 'lora_rank' in df.columns:
        ax3.scatter(df['lora_rank'], df['accuracy'], s=100, color=COLORS[1])
        for i, row in df.iterrows():
            ax3.annotate(row['name'], (row['lora_rank'], row['accuracy']),
                        textcoords="offset points", xytext=(5, 5))
        ax3.set_xlabel('LoRA Rank')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('LoRA Rank vs Accuracy')

    # 4. Tool별 정확도 히트맵
    ax4 = axes[1, 1]
    if 'tool_accuracy' in df.columns:
        tool_data = pd.DataFrame([exp.get('tool_accuracy', {}) for exp in experiments],
                                 index=df['name'])
        if not tool_data.empty:
            sns.heatmap(tool_data, annot=True, fmt='.1%', cmap='YlGnBu', ax=ax4)
            ax4.set_title('Tool-wise Accuracy Heatmap')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison chart saved to: {output_path}")


def plot_training_metrics(
    log_history: List[Dict],
    output_path: str
) -> None:
    """학습 메트릭 종합 플롯

    Args:
        log_history: 학습 로그 히스토리
        output_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss 추출
    train_data = [(x.get('step'), x.get('loss'), x.get('learning_rate'))
                  for x in log_history if 'loss' in x]
    eval_data = [(x.get('step'), x.get('eval_loss')) for x in log_history if 'eval_loss' in x]

    # 1. Train/Eval Loss 비교
    ax1 = axes[0, 0]
    if train_data:
        steps, losses, _ = zip(*train_data)
        ax1.plot(steps, losses, label='Train', color=COLORS[0], alpha=0.7)
    if eval_data:
        steps, losses = zip(*eval_data)
        ax1.plot(steps, losses, label='Eval', color=COLORS[1], linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Learning Rate Schedule
    ax2 = axes[0, 1]
    if train_data:
        steps, _, lrs = zip(*train_data)
        lrs = [lr for lr in lrs if lr is not None]
        if lrs:
            ax2.plot(range(len(lrs)), lrs, color=COLORS[2])
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)

    # 3. Loss 분포 (히스토그램)
    ax3 = axes[1, 0]
    if train_data:
        _, losses, _ = zip(*train_data)
        ax3.hist(losses, bins=30, color=COLORS[0], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Loss')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')

    # 4. Gradient (if available)
    ax4 = axes[1, 1]
    grad_data = [(x.get('step'), x.get('grad_norm')) for x in log_history if 'grad_norm' in x]
    if grad_data:
        steps, grads = zip(*grad_data)
        ax4.plot(steps, grads, color=COLORS[3])
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm over Training')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Gradient norm data not available',
                ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training metrics saved to: {output_path}")


def generate_all_visualizations(
    experiment_dir: str,
    log_file: str = "training_log.json",
    eval_file: str = "evaluation_results.json"
) -> None:
    """모든 시각화 생성

    Args:
        experiment_dir: 실험 디렉토리 경로
        log_file: 학습 로그 파일명
        eval_file: 평가 결과 파일명
    """
    print(f"\nGenerating visualizations for: {experiment_dir}")

    # 학습 로그 시각화
    log_path = os.path.join(experiment_dir, log_file)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_history = json.load(f)

        plot_loss_curve(
            log_history,
            os.path.join(experiment_dir, "loss_curve.png")
        )

        plot_training_metrics(
            log_history,
            os.path.join(experiment_dir, "training_metrics.png")
        )
    else:
        print(f"Warning: Log file not found: {log_path}")

    # 평가 결과 시각화
    eval_path = os.path.join(experiment_dir, eval_file)
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)

        plot_tool_accuracy(
            eval_results.get("tool_accuracy", {}),
            eval_results.get("tool_stats", {}),
            os.path.join(experiment_dir, "tool_accuracy.png")
        )

        plot_error_distribution(
            eval_results.get("error_distribution", {}),
            os.path.join(experiment_dir, "error_distribution.png")
        )
    else:
        print(f"Warning: Evaluation file not found: {eval_path}")

    print("Visualization generation completed!")


if __name__ == "__main__":
    # 테스트
    print("Visualization module test")

    # 샘플 데이터로 테스트
    sample_log = [
        {"step": 10, "loss": 1.5, "learning_rate": 2e-4},
        {"step": 20, "loss": 1.3, "learning_rate": 1.9e-4},
        {"step": 30, "loss": 1.1, "learning_rate": 1.8e-4},
        {"step": 30, "eval_loss": 1.2},
        {"step": 40, "loss": 0.9, "learning_rate": 1.7e-4},
        {"step": 50, "loss": 0.8, "learning_rate": 1.6e-4},
        {"step": 50, "eval_loss": 1.0},
    ]

    os.makedirs("results/test", exist_ok=True)
    plot_loss_curve(sample_log, "results/test/test_loss_curve.png", "Test Loss Curve")

    sample_tool_accuracy = {
        "predict_temperature": 0.85,
        "predict_yield": 0.68,
        "optimize_panel_angle": 0.52,
        "get_weather": 0.78,
    }
    sample_tool_stats = {
        "predict_temperature": {"correct": 17, "total": 20},
        "predict_yield": {"correct": 17, "total": 25},
        "optimize_panel_angle": {"correct": 13, "total": 25},
        "get_weather": {"correct": 23, "total": 30},
    }

    plot_tool_accuracy(
        sample_tool_accuracy,
        sample_tool_stats,
        "results/test/test_tool_accuracy.png"
    )

    print("Test visualizations created in results/test/")
