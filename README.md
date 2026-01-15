# Agentic AI R&D project

농업 도메인 특화 LLM 시스템 구축을 위한 R&D

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

이 레포지토리는 **LLM 파인튜닝 → 고성능 서빙 → RAG 최적화**까지 프로덕션 AI 시스템 구축에 필요한 기술을 R&D 하기 위해 만들어졌습니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI/ML Pipeline                                │
├─────────────────┬─────────────────────┬─────────────────────────────┤
│   Project 1     │     Project 2       │        Project 3            │
│  Fine-tuning    │   High-Performance  │     RAG Optimization        │
│                 │      Serving        │                             │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│  QLoRA 학습     │   vLLM Engine       │   Chunking 전략 비교        │
│  Tool Calling   │   Microservices     │   벡터 검색 최적화          │
│  실험 관리      │   Docker 배포       │   정량적 평가               │
└─────────────────┴─────────────────────┴─────────────────────────────┘
```

---

## Projects

### 🔬 Project 1: sLLM Fine-tuning for Tool Calling

> **농업 도메인 특화 Tool Calling을 위한 Qwen2.5-7B QLoRA 파인튜닝**

LLM이 외부 도구(API)를 정확하게 호출할 수 있도록 학습시키는 프로젝트입니다. 농업 분야의 수확량 예측, 기상 정보 조회, 태양광 패널 최적화 등 실제 도구 호출 시나리오를 구현합니다.

**핵심 기술:**
- **QLoRA (4-bit Quantization + LoRA)**: 16GB VRAM에서 7B 모델 학습
- **Tool Calling Format**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **체계적 실험 설계**: LoRA rank, learning rate, batch size 그리드 서치
- **정량적 평가**: Tool Calling Accuracy, Argument Extraction Accuracy

**실험 결과 (예상):**
| Config | LoRA Rank | LR | Tool Accuracy |
|--------|-----------|-----|---------------|
| exp001 | 16 | 1e-4 | 68.5% |
| exp002 | 32 | 1e-4 | 72.3% |
| **exp003** | **32** | **2e-4** | **76.8%** |

```
agri-sllm-finetuning/
├── notebooks/           # 학습 및 평가 노트북
├── src/                 # 데이터 생성, 학습, 평가 모듈
├── configs/             # 실험별 YAML 설정
└── results/             # 학습 로그 및 체크포인트
```

[📁 상세 보기](./agri-sllm-finetuning/)

---

### 🚀 Project 2: High-Performance LLM Serving with vLLM

> **vLLM 기반 고성능 추론 서버 + FastAPI 비즈니스 로직 분리 아키텍처**

프로덕션 환경에서 LLM을 효율적으로 서빙하기 위한 마이크로서비스 아키텍처를 구현합니다. vLLM의 Continuous Batching과 PagedAttention을 활용하여 처리량을 극대화합니다.

**핵심 기술:**
- **vLLM Engine**: Continuous Batching, PagedAttention, AWQ 4-bit 양자화
- **Microservices**: API Server(FastAPI) ↔ Inference Server(vLLM) 분리
- **Docker Compose**: 서비스 오케스트레이션, 헬스 체크, 그레이스풀 셧다운
- **성능 벤치마크**: Latency (P50/P95/P99), Throughput, GPU Utilization

**아키텍처:**
```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client     │────▶│   API Server     │────▶│ Inference Server│
│              │     │   (FastAPI)      │     │    (vLLM)       │
│              │     │   Port: 8000     │     │   Port: 8001    │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │                         │
                            ▼                         ▼
                     ┌─────────────┐          ┌─────────────┐
                     │ Prometheus  │          │    GPU      │
                     │  + Grafana  │          │   (CUDA)    │
                     └─────────────┘          └─────────────┘
```

**성능 지표 (예상):**
| Metric | Value |
|--------|-------|
| Throughput | 45 req/s |
| P50 Latency | 120ms |
| P95 Latency | 280ms |
| GPU Utilization | 85% |

```
vllm-deployment-study/
├── inference_server/    # vLLM 추론 서버
├── api_server/          # FastAPI 비즈니스 로직
├── benchmarks/          # 성능 테스트 스크립트
├── monitoring/          # Prometheus + Grafana
└── docker-compose.yml   # 서비스 오케스트레이션
```

[📁 상세 보기](./vllm-deployment-study/)

---

### 📊 Project 3: RAG Chunking Strategy Comparison

> **농업 문서에 최적화된 청킹 전략 도출을 위한 체계적 실험**

RAG 시스템의 핵심인 문서 청킹 전략을 비교 분석합니다. Fixed-size, Recursive, Semantic 세 가지 전략의 검색 성능을 정량적으로 평가하여 도메인 특성에 맞는 최적 전략을 제안합니다.

**핵심 기술:**
- **3가지 청킹 전략**: Fixed-size, Recursive (LangChain 방식), Semantic (임베딩 기반)
- **BGE-M3 임베딩**: 다국어 지원, Dense + Sparse 임베딩
- **FAISS 벡터 검색**: 고속 유사도 검색
- **정량적 평가**: Precision@k, Recall@k, F1@k, MRR

**실험 결과 (예상):**
| Strategy | Precision@5 | Recall@5 | F1@5 | Processing Time |
|----------|-------------|----------|------|-----------------|
| Fixed | 0.72 | 0.68 | 0.70 | **2.3s** |
| **Recursive** | **0.84** | **0.81** | **0.82** | 3.1s |
| Semantic | 0.78 | 0.75 | 0.76 | 45.2s |

**결론**: 농업 매뉴얼처럼 섹션 구조가 명확한 문서에는 **Recursive Chunking**이 최적

```
rag-chunking-experiments/
├── src/chunkers/        # 청킹 전략 구현
├── src/evaluators/      # 평가 메트릭
├── notebooks/           # 실험 노트북 (6개)
├── data/                # 샘플 문서 + 테스트 쿼리
└── results/             # 전략별 결과 비교
```

[📁 상세 보기](./rag-chunking-experiments/)

---

## Tech Stack

### Models & Frameworks
| Category | Technologies |
|----------|-------------|
| **Base Model** | Qwen2.5-7B-Instruct, Qwen2.5-7B-Instruct-AWQ |
| **Fine-tuning** | QLoRA, PEFT, TRL, bitsandbytes |
| **Serving** | vLLM, FastAPI, Uvicorn |
| **Embedding** | BAAI/bge-m3, Sentence-Transformers |
| **Vector DB** | FAISS |

### Infrastructure
| Category | Technologies |
|----------|-------------|
| **Container** | Docker, Docker Compose |
| **Monitoring** | Prometheus, Grafana |
| **GPU** | CUDA 12.1+, NVIDIA Container Toolkit |

### Development
| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML Framework** | PyTorch 2.0+ |
| **Experiment Tracking** | Weights & Biases (optional) |
| **Testing** | pytest, Locust |

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (권장)
- Docker & Docker Compose
- CUDA 12.1+ Driver

### Installation

```bash
# 레포지토리 클론
git clone <repository-url>
cd env-portfolio-v2

# 각 프로젝트별 환경 설정
# Project 1: Fine-tuning
cd agri-sllm-finetuning
pip install -r requirements.txt

# Project 2: Serving
cd ../vllm-deployment-study
docker-compose up -d

# Project 3: RAG
cd ../rag-chunking-experiments
pip install -r requirements.txt
python run_experiments.py --all
```

---

## Project Interconnections

세 프로젝트는 실제 AI 시스템 구축 파이프라인을 반영합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production AI System                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Project 1   │    │  Project 2   │    │  Project 3   │      │
│  │  Fine-tuned  │───▶│   Served     │◀───│  RAG-enabled │      │
│  │    Model     │    │   via vLLM   │    │   Context    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              User Query: "토마토 수확량 예측해줘"         │   │
│  │                                                          │   │
│  │  1. RAG: 관련 농업 문서 검색 (Project 3)                 │   │
│  │  2. Serving: vLLM으로 고속 추론 (Project 2)              │   │
│  │  3. Tool Calling: predict_yield 도구 호출 (Project 1)   │   │
│  │  4. Response: "예상 수확량은 3.2톤/ha입니다"             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Learnings & Highlights

### 🎯 실무 연관성

1. **메모리 효율적 학습**: QLoRA로 16GB GPU에서 7B 모델 파인튜닝
2. **프로덕션 아키텍처**: API/추론 분리로 독립적 스케일링 가능
3. **정량적 의사결정**: 실험 기반으로 최적 전략 선택

### 📈 기술적 깊이

- vLLM의 PagedAttention 동작 원리 이해
- Tool Calling 포맷 설계 및 파싱 구현
- Retrieval 평가 메트릭 (Precision, Recall, MRR) 직접 구현

### 📝 문서화

- 각 프로젝트별 상세 README
- 실험 방법론 및 결과 분석 문서
- 재현 가능한 실험 설정 (YAML configs)

---

## Repository Structure

```
env-portfolio-v2/
│
├── README.md                      # 이 파일
├── .gitignore
│
├── agri-sllm-finetuning/          # Project 1: Fine-tuning
│   ├── notebooks/
│   ├── src/
│   ├── configs/
│   └── docs/
│
├── vllm-deployment-study/         # Project 2: Serving
│   ├── inference_server/
│   ├── api_server/
│   ├── benchmarks/
│   └── monitoring/
│
└── rag-chunking-experiments/      # Project 3: RAG
    ├── src/
    ├── notebooks/
    ├── data/
    └── results/
```
