"""
Tool Calling 평가 모듈

이 모듈은 fine-tuned 모델의 Tool Calling 정확도를 평가합니다.
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from .data_utils import load_config, SYSTEM_PROMPT, AVAILABLE_TOOLS


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    case_id: str
    user_query: str
    expected_tool: Optional[str]
    expected_args: Optional[Dict]
    predicted_tool: Optional[str]
    predicted_args: Optional[Dict]
    is_correct: bool
    error_type: Optional[str]
    raw_response: str


def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """모델 응답에서 Tool Call 추출

    Args:
        response: 모델 응답 텍스트

    Returns:
        파싱된 tool call 딕셔너리 또는 None
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None

    try:
        tool_call_json = match.group(1).strip()
        tool_call = json.loads(tool_call_json)
        return tool_call
    except json.JSONDecodeError:
        return None


def compare_tool_calls(
    predicted: Optional[Dict],
    expected: Optional[Dict]
) -> Tuple[bool, Optional[str]]:
    """Tool Call 비교

    Args:
        predicted: 예측된 tool call
        expected: 기대되는 tool call

    Returns:
        (일치 여부, 오류 타입)
    """
    # 둘 다 None인 경우 (Tool 호출 안 함이 정답)
    if predicted is None and expected is None:
        return True, None

    # 예측은 None인데 기대값이 있는 경우
    if predicted is None and expected is not None:
        return False, "no_tool_call"

    # 예측은 있는데 기대값이 None인 경우
    if predicted is not None and expected is None:
        return False, "unexpected_tool_call"

    # Tool 이름 비교
    if predicted.get("name") != expected.get("name"):
        return False, "wrong_tool"

    # Arguments 비교
    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})

    # 필수 파라미터 확인 (expected에 있는 것들)
    for key, value in exp_args.items():
        if key not in pred_args:
            return False, "missing_argument"

        # 값 비교 (타입 변환 고려)
        pred_value = pred_args[key]

        # 문자열 비교 (대소문자 무시, 공백 제거)
        if isinstance(value, str) and isinstance(pred_value, str):
            if value.strip().lower() != pred_value.strip().lower():
                return False, "wrong_argument_value"
        elif value != pred_value:
            return False, "wrong_argument_value"

    return True, None


def load_model_for_inference(
    base_model_name: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """추론용 모델 로드

    Args:
        base_model_name: 베이스 모델 이름
        checkpoint_path: LoRA 체크포인트 경로
        device: 디바이스

    Returns:
        (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")

    # BitsAndBytes 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 어댑터 로드
    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_query: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """모델 응답 생성

    Args:
        model: 모델
        tokenizer: 토크나이저
        user_query: 사용자 질문
        system_prompt: 시스템 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도

    Returns:
        생성된 응답
    """
    # 프롬프트 구성
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 디코딩
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Assistant 응답만 추출
    response = full_response.split("<|im_start|>assistant\n")[-1]
    response = response.split("<|im_end|>")[0]

    return response.strip()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_cases: List[Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """모델 평가 실행

    Args:
        model: 모델
        tokenizer: 토크나이저
        test_cases: 테스트 케이스 리스트
        verbose: 상세 출력 여부

    Returns:
        평가 결과 딕셔너리
    """
    results = []
    correct = 0
    total = len(test_cases)

    # 오류 타입별 카운트
    error_counts = {
        "no_tool_call": 0,
        "unexpected_tool_call": 0,
        "wrong_tool": 0,
        "missing_argument": 0,
        "wrong_argument_value": 0,
    }

    # Tool별 정확도
    tool_stats = {}

    iterator = tqdm(test_cases, desc="Evaluating") if verbose else test_cases

    for case in iterator:
        case_id = case["id"]
        user_query = case["user_query"]
        expected_tool_call = case.get("expected_tool_call")

        # 응답 생성
        response = generate_response(model, tokenizer, user_query)

        # Tool call 추출
        predicted_tool_call = extract_tool_call(response)

        # 비교
        is_correct, error_type = compare_tool_calls(predicted_tool_call, expected_tool_call)

        if is_correct:
            correct += 1
        elif error_type:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Tool별 통계
        expected_tool = expected_tool_call.get("name") if expected_tool_call else "none"
        if expected_tool not in tool_stats:
            tool_stats[expected_tool] = {"correct": 0, "total": 0}
        tool_stats[expected_tool]["total"] += 1
        if is_correct:
            tool_stats[expected_tool]["correct"] += 1

        # 결과 저장
        result = EvaluationResult(
            case_id=case_id,
            user_query=user_query,
            expected_tool=expected_tool_call.get("name") if expected_tool_call else None,
            expected_args=expected_tool_call.get("arguments") if expected_tool_call else None,
            predicted_tool=predicted_tool_call.get("name") if predicted_tool_call else None,
            predicted_args=predicted_tool_call.get("arguments") if predicted_tool_call else None,
            is_correct=is_correct,
            error_type=error_type,
            raw_response=response,
        )
        results.append(result)

    # 정확도 계산
    accuracy = correct / total if total > 0 else 0

    # Tool별 정확도 계산
    tool_accuracy = {}
    for tool, stats in tool_stats.items():
        tool_accuracy[tool] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "error_distribution": error_counts,
        "tool_accuracy": tool_accuracy,
        "tool_stats": tool_stats,
        "detailed_results": [asdict(r) for r in results],
    }


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """평가 메트릭 계산

    Args:
        results: 평가 결과

    Returns:
        메트릭 딕셔너리
    """
    detailed = results["detailed_results"]

    # 기본 메트릭
    metrics = {
        "accuracy": results["accuracy"],
        "total_samples": results["total"],
        "correct_samples": results["correct"],
    }

    # Tool별 메트릭
    for tool, acc in results["tool_accuracy"].items():
        metrics[f"accuracy_{tool}"] = acc

    # 오류율
    total_errors = sum(results["error_distribution"].values())
    for error_type, count in results["error_distribution"].items():
        metrics[f"error_rate_{error_type}"] = count / results["total"] if results["total"] > 0 else 0

    return metrics


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """평가 결과 저장

    Args:
        results: 평가 결과
        output_path: 출력 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Evaluate Tool Calling Accuracy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/evaluation/tool_calling_test_cases.json",
        help="Path to test cases JSON"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results"
    )

    args = parser.parse_args()

    # 테스트 데이터 로드
    print(f"Loading test data from: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print(f"Total test cases: {len(test_cases)}")

    # 모델 로드
    model, tokenizer = load_model_for_inference(
        args.base_model,
        args.checkpoint
    )

    # 평가 실행
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60 + "\n")

    results = evaluate_model(model, tokenizer, test_cases)

    # 결과 출력
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    print("\nTool-wise Accuracy:")
    for tool, acc in results["tool_accuracy"].items():
        stats = results["tool_stats"][tool]
        print(f"  {tool}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print("\nError Distribution:")
    for error_type, count in results["error_distribution"].items():
        if count > 0:
            print(f"  {error_type}: {count}")

    # 결과 저장
    if args.output:
        output_path = args.output
    else:
        # checkpoint 경로에서 experiment 이름 추출
        exp_name = os.path.basename(os.path.dirname(args.checkpoint))
        output_path = f"results/{exp_name}/evaluation_results.json"

    save_evaluation_results(results, output_path)

    # 메트릭 계산 및 출력
    metrics = calculate_metrics(results)
    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
