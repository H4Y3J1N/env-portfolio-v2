# 데이터 설계 문서

## 개요

이 문서는 농업 도메인 Tool Calling을 위한 Instruction 데이터의 설계 및 구조를 설명합니다.

## 데이터셋 구조

### 디렉토리 구조

```
data/
├── raw/
│   └── agriculture_manual_excerpts.txt    # 농업 매뉴얼 원본
├── processed/
│   ├── instruction_data_sample.json       # 전체 학습 데이터
│   ├── train.json                         # 학습용 분할 (90%)
│   └── val.json                           # 검증용 분할 (10%)
└── evaluation/
    └── tool_calling_test_cases.json       # 평가용 테스트 케이스
```

## Tool 정의

### 1. predict_yield (수확량 예측)

```json
{
    "name": "predict_yield",
    "description": "작물의 예상 수확량을 예측합니다",
    "parameters": {
        "type": "object",
        "properties": {
            "crop_type": {
                "type": "string",
                "description": "작물 종류",
                "enum": ["토마토", "고추", "상추", "배추", "오이"]
            },
            "area_m2": {
                "type": "number",
                "description": "재배 면적 (제곱미터)"
            },
            "planting_date": {
                "type": "string",
                "description": "파종일 (YYYY-MM-DD)"
            }
        },
        "required": ["crop_type", "area_m2"]
    }
}
```

### 2. predict_temperature (기온 예측)

```json
{
    "name": "predict_temperature",
    "description": "특정 날짜의 기온을 예측합니다",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "예측 날짜 (YYYY-MM-DD)"
            },
            "location": {
                "type": "string",
                "description": "지역명"
            }
        },
        "required": ["date", "location"]
    }
}
```

### 3. optimize_panel_angle (태양광 패널 각도 최적화)

```json
{
    "name": "optimize_panel_angle",
    "description": "태양광 패널의 최적 각도를 계산합니다",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "위도"
            },
            "month": {
                "type": "integer",
                "description": "월 (1-12)"
            },
            "panel_type": {
                "type": "string",
                "description": "패널 유형",
                "enum": ["fixed", "tracking"]
            }
        },
        "required": ["latitude", "month"]
    }
}
```

### 4. get_weather (날씨 정보 조회)

```json
{
    "name": "get_weather",
    "description": "특정 지역의 날씨 정보를 조회합니다",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "지역명"
            },
            "date": {
                "type": "string",
                "description": "조회 날짜 (YYYY-MM-DD)"
            }
        },
        "required": ["location"]
    }
}
```

## Instruction 데이터 포맷

### JSON 구조

```json
{
    "id": "sample_001",
    "conversations": [
        {
            "role": "system",
            "content": "당신은 농업 전문 AI 어시스턴트입니다..."
        },
        {
            "role": "user",
            "content": "사용자 질문"
        },
        {
            "role": "assistant",
            "content": "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"
        }
    ],
    "metadata": {
        "tool_used": "tool_name",
        "complexity": "simple|medium|complex",
        "domain": "yield|weather|solar|general"
    }
}
```

### Tool Call 형식

모델이 생성해야 할 Tool Call 형식:

```
<tool_call>{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}</tool_call>
```

**주의사항:**
- JSON은 반드시 유효한 형식이어야 함
- `name`과 `arguments` 필드 필수
- arguments 내 필수 파라미터 포함 필요

## Qwen2.5 Chat Format

학습 데이터는 Qwen2.5의 Chat Format으로 변환됩니다:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

### 변환 예시

**원본 데이터:**
```json
{
    "role": "user",
    "content": "토마토 100평 재배하면 수확량이 얼마나 될까요?"
}
```

**변환 후:**
```
<|im_start|>user
토마토 100평 재배하면 수확량이 얼마나 될까요?<|im_end|>
```

## 데이터 복잡도 분류

### Simple (단순)
- 단일 Tool 호출
- 명확한 파라미터
- 직접적인 질문

예시:
> "서울 내일 날씨 알려줘"
> → `get_weather(location="서울", date="YYYY-MM-DD")`

### Medium (중간)
- 단일 Tool 호출
- 추론이 필요한 파라미터
- 간접적인 표현

예시:
> "이번 달에 태양광 패널 각도를 어떻게 설정하면 좋을까요? 위도는 37도입니다."
> → `optimize_panel_angle(latitude=37, month=현재월)`

### Complex (복잡)
- 다중 Tool 호출 가능성
- 복합적인 정보 요청
- 맥락 이해 필요

예시:
> "내일 서울 날씨와 이번 달 태양광 패널 최적 각도를 알려주세요."
> → `get_weather()` + `optimize_panel_angle()`

## 데이터 분할

### 비율
- Train: 90%
- Validation: 10%

### 분할 기준
- 층화 샘플링 (Stratified Sampling) by tool_used
- 각 Tool에 대한 균등한 분포 유지
- Random seed: 42

## 데이터 검증

### 필수 검증 항목

1. **구조 검증**
   - `id` 필드 존재
   - `conversations` 배열 존재
   - 최소 2개 이상의 메시지 (user + assistant)

2. **Tool Call 검증**
   - `<tool_call>` 태그 존재
   - 유효한 JSON 형식
   - `name` 필드 존재
   - `arguments` 필드 존재
   - 필수 파라미터 포함

3. **메타데이터 검증**
   - `tool_used` 필드 유효
   - `complexity` 값 유효

### 검증 코드

```python
from src.data_utils import validate_conversation, validate_tool_call

# 대화 구조 검증
is_valid, error_msg = validate_conversation(conversation)

# Tool Call 형식 검증
is_valid, error_msg, parsed = validate_tool_call(tool_call_str)
```

## 데이터 확장 가이드

### 새로운 샘플 추가 시

1. JSON 형식 준수
2. Tool Call 형식 정확히 작성
3. metadata 포함
4. 검증 스크립트 실행

### 새로운 Tool 추가 시

1. `src/data_utils.py`의 `AVAILABLE_TOOLS`에 추가
2. Tool 정의 (name, description, parameters)
3. 해당 Tool 사용 샘플 최소 10개 이상 작성
4. 테스트 케이스 추가

## 참고 사항

- 모든 날짜는 ISO 8601 형식 (YYYY-MM-DD)
- 한국어 데이터 위주
- 농업 도메인 특화 용어 사용
- 실제 농업 상황을 반영한 자연스러운 대화
