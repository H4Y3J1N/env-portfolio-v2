# Tool Calling 평가 방법론

## 개요

이 문서는 Fine-tuning된 모델의 Tool Calling 능력을 평가하기 위한 방법론을 설명합니다.

## 평가 목표

1. **Tool 선택 정확도**: 올바른 Tool을 선택하는지 평가
2. **Argument 정확도**: 올바른 인자를 추출하는지 평가
3. **형식 준수**: Tool Call 형식을 올바르게 생성하는지 평가
4. **복잡도별 성능**: 난이도에 따른 성능 변화 분석

## 평가 메트릭

### 1. Overall Accuracy

전체 테스트 케이스 중 완전히 정확한 Tool Call의 비율

```
Overall Accuracy = (정확한 예측 수) / (전체 테스트 케이스 수) × 100%
```

### 2. Tool Selection Accuracy

올바른 Tool을 선택한 비율 (인자와 무관)

```
Tool Selection Accuracy = (올바른 Tool 선택 수) / (전체 테스트 케이스 수) × 100%
```

### 3. Argument Accuracy

Tool이 정확한 경우, 인자도 정확한 비율

```
Argument Accuracy = (Tool+Argument 모두 정확) / (Tool이 정확한 케이스) × 100%
```

### 4. Format Compliance Rate

유효한 Tool Call 형식을 생성한 비율

```
Format Compliance = (유효한 형식) / (전체 응답) × 100%
```

## 에러 유형 분류

### 1. no_tool_call
- **정의**: 응답에 `<tool_call>` 태그가 없음
- **원인**: 모델이 Tool Call이 필요하다고 판단하지 않음
- **심각도**: High

### 2. invalid_format
- **정의**: Tool Call 형식이 잘못됨 (JSON 파싱 실패)
- **원인**: 모델이 올바른 형식을 학습하지 못함
- **심각도**: High

### 3. wrong_tool
- **정의**: 잘못된 Tool을 선택함
- **원인**: Tool 선택 학습 부족
- **심각도**: High

### 4. missing_argument
- **정의**: 필수 인자가 누락됨
- **원인**: 인자 추출 학습 부족
- **심각도**: Medium

### 5. wrong_argument_value
- **정의**: 인자 값이 잘못됨
- **원인**: 값 추론 학습 부족
- **심각도**: Medium

### 6. extra_argument
- **정의**: 불필요한 인자가 포함됨
- **원인**: 과잉 생성
- **심각도**: Low

## 테스트 케이스 설계

### 테스트 케이스 구조

```json
{
    "id": "test_001",
    "query": "사용자 질문",
    "expected_tool": "예상 Tool 이름",
    "expected_arguments": {
        "param1": "value1",
        "param2": "value2"
    },
    "complexity": "simple|medium|complex",
    "category": "yield|weather|solar|general"
}
```

### 복잡도별 테스트 케이스 분포

| 복잡도 | 비율 | 설명 |
|--------|------|------|
| Simple | 40% | 명확한 단일 Tool 호출 |
| Medium | 40% | 추론이 필요한 케이스 |
| Complex | 20% | 복합적인 요청 |

### Tool별 테스트 케이스 분포

| Tool | 비율 |
|------|------|
| predict_yield | 25% |
| predict_temperature | 25% |
| optimize_panel_angle | 25% |
| get_weather | 25% |

## 평가 프로세스

### Step 1: 응답 생성

```python
response = generate_response(model, tokenizer, query)
```

- Temperature: 0.1 (일관성 위해 낮게 설정)
- Max tokens: 512
- Sampling: True

### Step 2: Tool Call 추출

```python
from src.evaluation import extract_tool_call

predicted = extract_tool_call(response)
# Returns: {"name": "tool_name", "arguments": {...}} or None
```

### Step 3: 비교 및 판정

```python
from src.evaluation import compare_tool_calls

is_correct, error_type = compare_tool_calls(predicted, expected)
```

**비교 로직:**

1. `predicted`가 None → `no_tool_call`
2. `predicted`가 Dict이 아님 → `invalid_format`
3. `name`이 다름 → `wrong_tool`
4. 필수 argument 누락 → `missing_argument`
5. argument 값이 다름 → `wrong_argument_value`
6. 모두 일치 → `correct`

### Step 4: 메트릭 계산

```python
from src.evaluation import calculate_metrics

metrics = calculate_metrics(results)
# Returns: {
#     "overall_accuracy": float,
#     "tool_accuracy": float,
#     "argument_accuracy": float,
#     "format_compliance": float,
#     "error_distribution": {...}
# }
```

## 평가 기준

### 정확성 판정 기준

| 항목 | 기준 |
|------|------|
| Tool Name | 완전 일치 |
| String Argument | 완전 일치 또는 정규화 후 일치 |
| Number Argument | 완전 일치 |
| Date Argument | ISO 8601 형식으로 정규화 후 일치 |

### 정규화 규칙

1. **문자열**: 앞뒤 공백 제거, 소문자 변환
2. **숫자**: 정수/실수 타입 무관
3. **날짜**: YYYY-MM-DD 형식으로 정규화

## 결과 해석

### 목표 정확도

| 실험 단계 | 목표 정확도 |
|-----------|-------------|
| Baseline | 70%+ |
| Optimized | 80%+ |
| Production-ready | 90%+ |

### 성능 분석 관점

1. **전체 정확도**: 모델의 전반적 Tool Calling 능력
2. **Tool별 정확도**: 특정 Tool에 대한 강약점 파악
3. **복잡도별 정확도**: 난이도에 따른 성능 변화
4. **에러 분포**: 주요 개선 포인트 파악

### 개선 전략

| 에러 유형 | 개선 전략 |
|-----------|-----------|
| no_tool_call | Tool Call 필요성 판단 학습 강화 |
| invalid_format | 형식 학습 데이터 증강 |
| wrong_tool | Tool 선택 다양성 증가 |
| missing_argument | 인자 추출 예제 증강 |
| wrong_argument_value | 값 추론 예제 증강 |

## 보고서 생성

### 평가 요약 (JSON)

```json
{
    "experiment": "exp-001",
    "total_test_cases": 50,
    "correct_count": 38,
    "overall_accuracy": 76.0,
    "accuracy_by_tool": {...},
    "accuracy_by_complexity": {...},
    "error_distribution": {...}
}
```

### 시각화

1. **Tool별 정확도 바 차트**: 각 Tool의 성능 비교
2. **복잡도별 히트맵**: 실험/복잡도 조합별 성능
3. **에러 분포 파이 차트**: 에러 유형별 비율
4. **실험 비교 그래프**: 여러 실험의 성능 추이

## 재현성

### 실험 환경 기록

- Python 버전
- 주요 라이브러리 버전
- GPU 정보
- Random seed

### 결과 저장

모든 평가 결과는 다음 위치에 저장:

```
checkpoints/exp-XXX/
├── evaluation_summary.json    # 요약 메트릭
├── evaluation_detailed.json   # 상세 결과
└── evaluation_results.png     # 시각화
```

## 한계점 및 주의사항

1. **테스트 데이터 편향**: 학습 데이터와 유사한 분포
2. **정적 평가**: 실시간 API 호출 미포함
3. **단일 턴 평가**: Multi-turn 대화 미지원
4. **언어 제한**: 한국어 위주 평가

## 참고 자료

- [Tool Use in LLMs](https://arxiv.org/abs/2304.08354)
- [Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling)
- [Evaluation Metrics for NLG](https://aclanthology.org/2020.emnlp-main.448/)
