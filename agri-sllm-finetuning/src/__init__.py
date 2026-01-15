"""
농업 도메인 특화 sLLM Fine-tuning 프로젝트

이 패키지는 Qwen2.5-7B 모델을 농업 도메인 Tool Calling에
특화시키기 위한 QLoRA Fine-tuning 도구를 제공합니다.
"""

from .data_utils import (
    load_config,
    load_instruction_data,
    prepare_datasets,
    format_conversation,
    validate_tool_call,
)

from .training import (
    setup_model_and_tokenizer,
    setup_lora_config,
    create_trainer,
    train_model,
)

from .evaluation import (
    extract_tool_call,
    compare_tool_calls,
    evaluate_model,
    calculate_metrics,
)

from .visualization import (
    plot_loss_curve,
    plot_tool_accuracy,
    plot_error_distribution,
    create_comparison_chart,
)

__version__ = "1.0.0"
__author__ = "Agriculture AI Team"
