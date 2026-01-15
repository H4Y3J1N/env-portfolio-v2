# 실험 로그

## 개요

이 문서는 농업 도메인 Tool Calling sLLM Fine-tuning 프로젝트의 실험 로그를 기록합니다.

## 실험 설계

### Base Model
- **모델**: Qwen2.5-7B-Instruct
- **양자화**: 4-bit (NF4)
- **Fine-tuning 방식**: QLoRA (Quantized Low-Rank Adaptation)

### 공통 설정
| 항목 | 값 |
|------|-----|
| Quantization Type | nf4 |
| Compute Dtype | bfloat16 |
| Double Quantization | True |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA Dropout | 0.05 |
| Max Sequence Length | 2048 |
| Optimizer | paged_adamw_8bit |
| LR Scheduler | cosine |

---

## 실험 1: Baseline (exp-001)

### 목표
QLoRA Fine-tuning의 기본 성능 확인

### 설정
| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Epochs | 3 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Warmup Steps | 50 |

### 예상 결과
- Train Loss: ~0.5
- Eval Loss: ~0.6
- Tool Calling 정확도: ~70-75%

### 실행
```bash
# Jupyter 노트북에서 실행
EXPERIMENT = "exp-001"
```

### 결과
*(실험 후 기록)*

| 메트릭 | 값 |
|--------|-----|
| Final Train Loss | - |
| Final Eval Loss | - |
| Tool Calling Accuracy | - |

### 관찰 및 분석
*(실험 후 기록)*

---

## 실험 2: Learning Rate 최적화 (exp-002)

### 목표
낮은 Learning Rate와 더 많은 Epoch으로 안정적 학습 확인

### 변경 사항
- Learning Rate: 2e-4 → **1e-4** (50% 감소)
- Epochs: 3 → **5** (67% 증가)
- Batch Size: 4 → **8** (2배 증가)
- Gradient Accumulation: 4 → **2**

### 설정
| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Epochs | 5 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Warmup Steps | 100 |

### 가설
- 낮은 LR로 더 안정적인 수렴 예상
- 더 많은 Epoch으로 fine-grained optimization 기대
- 큰 Batch Size로 gradient 안정화

### 실행
```bash
EXPERIMENT = "exp-002"
```

### 결과
*(실험 후 기록)*

| 메트릭 | 값 |
|--------|-----|
| Final Train Loss | - |
| Final Eval Loss | - |
| Tool Calling Accuracy | - |

### 관찰 및 분석
*(실험 후 기록)*

---

## 실험 3: LoRA Rank 증가 (exp-003)

### 목표
더 높은 LoRA Rank로 모델 표현력 향상 확인

### 변경 사항
- LoRA Rank: 16 → **32** (2배 증가)
- LoRA Alpha: 32 → **64** (비율 유지)

### 설정
| 파라미터 | 값 |
|----------|-----|
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Epochs | 5 |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64 |
| Warmup Steps | 100 |

### 가설
- 높은 Rank로 더 복잡한 패턴 학습 가능
- Tool Calling의 다양한 케이스 처리 능력 향상 기대
- 학습 파라미터 수 증가로 VRAM 사용량 증가 예상

### 실행
```bash
EXPERIMENT = "exp-003"
```

### 결과
*(실험 후 기록)*

| 메트릭 | 값 |
|--------|-----|
| Final Train Loss | - |
| Final Eval Loss | - |
| Tool Calling Accuracy | - |

### 관찰 및 분석
*(실험 후 기록)*

---

## 실험 비교 요약

### 설정 비교
| 실험 | LR | Batch | Epochs | LoRA r | LoRA α |
|------|-----|-------|--------|--------|--------|
| exp-001 | 2e-4 | 4 | 3 | 16 | 32 |
| exp-002 | 1e-4 | 8 | 5 | 16 | 32 |
| exp-003 | 1e-4 | 8 | 5 | 32 | 64 |

### 결과 비교
*(실험 후 기록)*

| 실험 | Train Loss | Eval Loss | Accuracy |
|------|------------|-----------|----------|
| exp-001 | - | - | - |
| exp-002 | - | - | - |
| exp-003 | - | - | - |

---

## 결론 및 향후 계획

### 주요 발견
*(실험 후 기록)*

### 권장 설정
*(실험 후 기록)*

### 향후 실험 계획
1. 더 큰 데이터셋으로 재실험
2. 다른 base model 테스트 (Llama 3.1, Mistral 등)
3. Multi-tool calling 케이스 확대
4. 추가 Hyperparameter 탐색 (LoRA Rank 64, Alpha 128 등)

---

## 참고 자료

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
