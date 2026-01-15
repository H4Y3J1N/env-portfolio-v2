"""
데이터 로딩 및 전처리 유틸리티

이 모듈은 instruction 데이터의 로딩, 검증, 전처리 기능을 제공합니다.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import yaml
from sklearn.model_selection import train_test_split


@dataclass
class Tool:
    """Tool 정의 데이터 클래스"""
    name: str
    description: str
    parameters: Dict[str, Any]


# 사용 가능한 Tools 정의
AVAILABLE_TOOLS = {
    "predict_yield": Tool(
        name="predict_yield",
        description="작물 수확량 예측",
        parameters={
            "crop_type": {"type": "string", "required": True, "description": "작물 종류"},
            "area_sqm": {"type": "number", "required": True, "description": "재배 면적 (제곱미터)"},
            "planting_date": {"type": "string", "required": False, "description": "파종일 (YYYY-MM-DD)"},
        }
    ),
    "predict_temperature": Tool(
        name="predict_temperature",
        description="특정 시점의 기온 예측",
        parameters={
            "datetime": {"type": "string", "required": True, "description": "예측 시점 (ISO 8601)"},
            "location": {"type": "string", "required": False, "description": "위치 (기본: default_farm)"},
        }
    ),
    "optimize_panel_angle": Tool(
        name="optimize_panel_angle",
        description="태양광 패널 최적 각도 계산",
        parameters={
            "latitude": {"type": "number", "required": True, "description": "위도"},
            "longitude": {"type": "number", "required": True, "description": "경도"},
            "date": {"type": "string", "required": False, "description": "날짜 (YYYY-MM-DD)"},
            "time_of_day": {"type": "string", "required": False, "description": "시간대 (morning/noon/afternoon)"},
        }
    ),
    "get_weather": Tool(
        name="get_weather",
        description="현재 또는 예보 기상 정보 조회",
        parameters={
            "location": {"type": "string", "required": False, "description": "위치"},
            "date": {"type": "string", "required": False, "description": "날짜 (YYYY-MM-DD)"},
        }
    ),
}


SYSTEM_PROMPT = """당신은 영농형 태양광 전문 AI 어시스턴트입니다. 다음 도구들을 사용할 수 있습니다:

1. predict_yield: 작물 수확량 예측
   - 필수: crop_type (작물 종류), area_sqm (면적, 제곱미터)
   - 선택: planting_date (파종일)

2. predict_temperature: 기온 예측
   - 필수: datetime (예측 시점, ISO 8601 형식)
   - 선택: location (위치)

3. optimize_panel_angle: 태양광 패널 각도 최적화
   - 필수: latitude (위도), longitude (경도)
   - 선택: date (날짜), time_of_day (시간대)

4. get_weather: 기상 정보 조회
   - 선택: location (위치), date (날짜)

도구를 사용할 때는 반드시 다음 형식을 따르세요:
<tool_call>{"name": "도구이름", "arguments": {...}}</tool_call>

사용자의 질문에 적절한 도구를 선택하여 정확하게 호출하세요."""


def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_instruction_data(file_path: str) -> List[Dict[str, Any]]:
    """Instruction 데이터 로드

    Args:
        file_path: JSON 파일 경로

    Returns:
        instruction 데이터 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_instruction_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Instruction 데이터 저장

    Args:
        data: 저장할 데이터
        file_path: 저장 경로
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_tool_call(tool_call_str: str) -> Tuple[bool, str, Optional[Dict]]:
    """Tool call 문자열 검증

    Args:
        tool_call_str: <tool_call>...</tool_call> 형식의 문자열

    Returns:
        (유효 여부, 오류 메시지, 파싱된 tool call)
    """
    # 패턴 매칭
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, tool_call_str, re.DOTALL)

    if not match:
        return False, "No tool_call tag found", None

    try:
        tool_call = json.loads(match.group(1).strip())
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None

    # 필수 필드 확인
    if "name" not in tool_call:
        return False, "Missing 'name' field", None

    if "arguments" not in tool_call:
        return False, "Missing 'arguments' field", None

    # Tool 존재 확인
    tool_name = tool_call["name"]
    if tool_name not in AVAILABLE_TOOLS:
        return False, f"Unknown tool: {tool_name}", None

    # 필수 파라미터 확인
    tool = AVAILABLE_TOOLS[tool_name]
    for param_name, param_info in tool.parameters.items():
        if param_info.get("required", False):
            if param_name not in tool_call["arguments"]:
                return False, f"Missing required parameter: {param_name}", None

    return True, "Valid", tool_call


def validate_conversation(conversation: Dict[str, Any]) -> Tuple[bool, str]:
    """대화 구조 검증

    Args:
        conversation: 대화 데이터

    Returns:
        (유효 여부, 오류 메시지)
    """
    required_keys = ["id", "conversations"]

    for key in required_keys:
        if key not in conversation:
            return False, f"Missing key: {key}"

    messages = conversation["conversations"]

    # 최소한 system, user, assistant가 있어야 함
    roles = [msg.get("role") for msg in messages]

    if "system" not in roles:
        return False, "Missing system message"
    if "user" not in roles:
        return False, "Missing user message"
    if "assistant" not in roles:
        return False, "Missing assistant message"

    # Assistant 메시지에서 tool_call 형식 검증
    for msg in messages:
        if msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", ""):
            is_valid, error_msg, _ = validate_tool_call(msg["content"])
            if not is_valid:
                return False, f"Invalid tool_call: {error_msg}"

    return True, "Valid"


def format_conversation(conversation: Dict[str, Any], include_system: bool = True) -> str:
    """대화를 Qwen2.5 형식으로 포맷팅

    Args:
        conversation: 대화 데이터
        include_system: 시스템 프롬프트 포함 여부

    Returns:
        포맷팅된 텍스트
    """
    text = ""

    for msg in conversation["conversations"]:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            if include_system:
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            text += f"<|im_start|>tool\n{content}<|im_end|>\n"

    return text


def prepare_datasets(
    data_path: str = "data/processed/instruction_data_sample.json",
    output_dir: str = "data/processed",
    test_size: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """데이터셋 준비 (Train/Validation 분할)

    Args:
        data_path: 원본 데이터 경로
        output_dir: 출력 디렉토리
        test_size: 검증 세트 비율
        seed: 랜덤 시드

    Returns:
        (train_data, val_data)
    """
    # 데이터 로드
    data = load_instruction_data(data_path)

    print(f"총 샘플 수: {len(data)}")

    # 검증
    valid_data = []
    invalid_count = 0

    for item in data:
        is_valid, error_msg = validate_conversation(item)
        if is_valid:
            # 텍스트 필드 추가
            item["text"] = format_conversation(item)
            valid_data.append(item)
        else:
            invalid_count += 1
            print(f"Invalid sample {item.get('id', 'unknown')}: {error_msg}")

    print(f"유효 샘플: {len(valid_data)}, 무효 샘플: {invalid_count}")

    # Tool별 분포 확인
    tool_counts = {}
    for item in valid_data:
        tool = item.get("metadata", {}).get("tool_used", "unknown")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"Tool 분포: {tool_counts}")

    # Stratified split (Tool별로 균등 분배)
    try:
        stratify_labels = [item.get("metadata", {}).get("tool_used", "unknown") for item in valid_data]
        train_data, val_data = train_test_split(
            valid_data,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=seed
        )
    except ValueError:
        # Stratify 불가능할 경우 일반 split
        train_data, val_data = train_test_split(
            valid_data,
            test_size=test_size,
            random_state=seed
        )

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    save_instruction_data(train_data, os.path.join(output_dir, "train.json"))
    save_instruction_data(val_data, os.path.join(output_dir, "val.json"))

    return train_data, val_data


def get_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """데이터셋 통계 계산

    Args:
        data: instruction 데이터 리스트

    Returns:
        통계 딕셔너리
    """
    stats = {
        "total_samples": len(data),
        "tool_distribution": {},
        "complexity_distribution": {},
        "avg_conversation_length": 0,
        "avg_tokens_estimate": 0,
    }

    total_conv_length = 0
    total_chars = 0

    for item in data:
        # Tool 분포
        tool = item.get("metadata", {}).get("tool_used", "unknown")
        stats["tool_distribution"][tool] = stats["tool_distribution"].get(tool, 0) + 1

        # 복잡도 분포
        complexity = item.get("metadata", {}).get("complexity", "unknown")
        stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1

        # 대화 길이
        conv_length = len(item.get("conversations", []))
        total_conv_length += conv_length

        # 텍스트 길이 (토큰 수 추정)
        text = item.get("text", format_conversation(item))
        total_chars += len(text)

    stats["avg_conversation_length"] = total_conv_length / len(data) if data else 0
    stats["avg_tokens_estimate"] = (total_chars / 4) / len(data) if data else 0  # 대략 4자당 1토큰

    return stats


if __name__ == "__main__":
    # 테스트
    print("데이터 유틸리티 테스트")

    # Tool call 검증 테스트
    test_tool_call = '<tool_call>{"name": "predict_temperature", "arguments": {"datetime": "2025-01-18T10:00:00"}}</tool_call>'
    is_valid, msg, parsed = validate_tool_call(test_tool_call)
    print(f"Tool call 검증: {is_valid}, {msg}")

    # 데이터셋 준비 (파일이 있는 경우)
    data_path = "data/processed/instruction_data_sample.json"
    if os.path.exists(data_path):
        train_data, val_data = prepare_datasets(data_path)
        stats = get_dataset_statistics(train_data)
        print(f"통계: {stats}")
