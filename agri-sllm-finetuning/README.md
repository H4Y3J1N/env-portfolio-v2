# 농업 도메인 특화 sLLM Fine-tuning 실험

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 프로젝트 개요

Qwen2.5-7B 모델을 농업 도메인에 특화시켜 **Tool Calling 정확도를 향상**시키는 QLoRA Fine-tuning 실험 프로젝트입니다.

### 목표
- 영농형 태양광 도메인의 4가지 Tool에 대한 정확한 호출 능력 학습
- Tool Calling Accuracy **70% 이상** 달성

### 사용 가능한 Tools
| Tool | 설명 | 난이도 |
|------|------|--------|
| `predict_yield` | 작물 수확량 예측 | Medium |
| `predict_temperature` | 기온 예측 | Easy |
| `optimize_panel_angle` | 태양광 패널 각도 최적화 | Hard |
| `get_weather` | 기상 정보 조회 | Easy |

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 HuggingFace 토큰 입력
```

### 2. 데이터 준비

```bash
# 노트북 실행 또는 스크립트 실행
python -c "from src.data_utils import prepare_datasets; prepare_datasets()"
```

### 3. 학습 실행

```bash
# 실험 1 실행 (Baseline)
python -m src.training --config configs/training_config_exp001.yaml

# 또는 노트북에서 실행
jupyter notebook notebooks/02_qlora_training.ipynb
```

### 4. 평가

```bash
python -m src.evaluation --checkpoint results/exp-001/final_checkpoint
```

## 프로젝트 구조

```
agri-sllm-finetuning/
├── notebooks/
│   ├── 01_data_preparation.ipynb    # 데이터 전처리 및 검증
│   ├── 02_qlora_training.ipynb      # QLoRA Fine-tuning 학습
│   ├── 03_evaluation.ipynb          # Tool Calling 평가
│   └── 04_analysis.ipynb            # 실험 결과 분석
├── data/
│   ├── raw/                         # 원본 농업 매뉴얼
│   ├── processed/                   # 전처리된 instruction 데이터
│   └── evaluation/                  # 평가용 테스트 케이스
├── configs/                         # 실험별 설정 파일
├── src/                             # 소스 코드
├── results/                         # 실험 결과 (체크포인트 제외)
└── docs/                            # 문서
```

## 실험 설계

### 실험 1: Baseline (exp-001)
- Learning Rate: 2e-4
- Batch Size: 4
- LoRA Rank: 16
- Epochs: 3

### 실험 2: Learning Rate 최적화 (exp-002)
- Learning Rate: **1e-4** (↓)
- Batch Size: **8** (↑)
- LoRA Rank: 16
- Epochs: **5** (↑)

### 실험 3: LoRA Rank 증가 (exp-003)
- Learning Rate: 1e-4
- Batch Size: 8
- LoRA Rank: **32** (↑)
- Epochs: 5

## 데이터셋

### Instruction 데이터 구조
```json
{
  "id": "agri_tool_001",
  "conversations": [
    {"role": "system", "content": "시스템 프롬프트..."},
    {"role": "user", "content": "다음 주 토요일 기온을 알려주세요."},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"predict_temperature\", ...}</tool_call>"}
  ]
}
```

### 데이터 분포
- 단일 Tool 호출: 200개
- 멀티 Tool 호출: 150개
- 에러 처리: 100개
- 거부 케이스: 50개
- **총합: 500개** (샘플 50개 공개)

## 평가 메트릭

- **Tool Calling Accuracy**: Tool 이름 + Arguments 정확도
- **Tool별 Accuracy**: 각 Tool에 대한 개별 정확도
- **Error Type Distribution**: 오류 유형별 분석

## 결과 요약

| 실험 | Learning Rate | LoRA Rank | Accuracy |
|------|---------------|-----------|----------|
| exp-001 | 2e-4 | 16 | 65.0% |
| exp-002 | 1e-4 | 16 | 71.0% |
| exp-003 | 1e-4 | 32 | 74.0% |

## 요구사항

- Python 3.10+
- CUDA 12.1+ (GPU 필수)
- VRAM 16GB+ (RTX 4090 권장)

## 라이선스

MIT License
