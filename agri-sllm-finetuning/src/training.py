"""
QLoRA Fine-tuning 학습 모듈

이 모듈은 Qwen2.5-7B 모델의 QLoRA fine-tuning 기능을 제공합니다.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import yaml

from .data_utils import load_config, format_conversation


class TrainingProgressCallback(TrainerCallback):
    """학습 진행 상황 출력 콜백"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                print(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: eval_loss = {logs['eval_loss']:.4f}")


def setup_model_and_tokenizer(
    config: Dict[str, Any]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """모델과 토크나이저 설정

    Args:
        config: 설정 딕셔너리

    Returns:
        (model, tokenizer)
    """
    model_config = config["model"]

    print(f"Loading model: {model_config['base_model']}")

    # BitsAndBytes 설정 (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.get("load_in_4bit", True),
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # Padding 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Model loaded successfully!")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """LoRA 설정 생성

    Args:
        config: 설정 딕셔너리

    Returns:
        LoraConfig 객체
    """
    lora_config = config["lora"]

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )

    return peft_config


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    peft_config: LoraConfig
) -> AutoModelForCausalLM:
    """모델을 학습용으로 준비

    Args:
        model: 베이스 모델
        peft_config: LoRA 설정

    Returns:
        PEFT 모델
    """
    # k-bit 학습 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 적용
    model = get_peft_model(model, peft_config)

    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    return model


def load_datasets(config: Dict[str, Any]):
    """데이터셋 로드

    Args:
        config: 설정 딕셔너리

    Returns:
        datasets 딕셔너리
    """
    data_config = config["data"]

    # JSON 데이터셋 로드
    dataset = load_dataset('json', data_files={
        'train': data_config["train_file"],
        'validation': data_config["val_file"],
    })

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    return dataset


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    config: Dict[str, Any]
) -> SFTTrainer:
    """SFTTrainer 생성

    Args:
        model: PEFT 모델
        tokenizer: 토크나이저
        dataset: 데이터셋
        config: 설정 딕셔너리

    Returns:
        SFTTrainer 객체
    """
    training_config = config["training"]
    misc_config = config["misc"]
    data_config = config["data"]
    optimizer_config = config["optimizer"]

    # 출력 디렉토리 생성
    output_dir = misc_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        eval_strategy="steps",
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        fp16=misc_config.get("fp16", False),
        bf16=misc_config.get("bf16", True),
        max_grad_norm=misc_config.get("max_grad_norm", 0.3),
        optim=optimizer_config.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=optimizer_config.get("lr_scheduler_type", "cosine"),
        seed=misc_config.get("seed", 42),
        report_to=misc_config.get("report_to", "none"),
        logging_dir=os.path.join(output_dir, "logs"),
    )

    # SFTTrainer 생성
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 2048),
        dataset_text_field=data_config.get("dataset_text_field", "text"),
        callbacks=[TrainingProgressCallback()],
    )

    return trainer


def train_model(config_path: str) -> None:
    """모델 학습 실행

    Args:
        config_path: 설정 파일 경로
    """
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    experiment_name = config["experiment"]["name"]
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Description: {config['experiment']['description']}")
    print(f"{'='*60}\n")

    # GPU 확인
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available. Training will be very slow!")

    # 모델 및 토크나이저 로드
    model, tokenizer = setup_model_and_tokenizer(config)

    # LoRA 설정
    peft_config = setup_lora_config(config)
    print(f"\nLoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # 모델 준비
    model = prepare_model_for_training(model, peft_config)

    # 데이터셋 로드
    dataset = load_datasets(config)

    # Trainer 생성
    trainer = create_trainer(model, tokenizer, dataset, config)

    # 학습 시작
    print("\n" + "="*60)
    print("Training started...")
    print("="*60 + "\n")

    trainer.train()

    # 모델 저장
    output_dir = config["misc"]["output_dir"]
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

    print(f"\nSaving model to: {final_checkpoint_dir}")
    trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)

    # 학습 로그 저장
    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log.json")

    import json
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)

    print(f"Training log saved to: {log_path}")
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Agriculture Domain")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )

    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
