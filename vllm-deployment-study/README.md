# vLLM Serving 아키텍처 실험

GPU 추론 서버와 API 서버를 분리한 프로덕션급 LLM Serving 아키텍처 설계 및 성능 벤치마크 프로젝트입니다.

## 프로젝트 개요

### 목표
- Docker Compose 기반 마이크로서비스 아키텍처 구축
- vLLM 고성능 추론 서버 구현
- FastAPI 기반 비즈니스 로직 API 서버 개발
- 성능 벤치마크 (Throughput, Latency, GPU Usage)

### 핵심 기술
- **vLLM**: Continuous Batching + PagedAttention으로 고성능 추론
- **FastAPI**: 비동기 API 서버
- **Docker Compose**: 마이크로서비스 오케스트레이션
- **Qwen2.5-7B-AWQ**: 4-bit 양자화 모델

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                      Client Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Web UI  │  │ Mobile   │  │  cURL    │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
└───────┼─────────────┼─────────────┼────────────────────┘
        └─────────────┴─────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              API Server (FastAPI)                        │
│  Port: 8000                                             │
│  - Request Validation                                   │
│  - Prompt Engineering                                   │
│  - Response Post-processing                             │
└────────────────────────┬────────────────────────────────┘
                         ↓ HTTP (Internal Network)
┌─────────────────────────────────────────────────────────┐
│         Inference Server (vLLM)                          │
│  Port: 8001 (Internal Only)                             │
│  - Continuous Batching                                  │
│  - PagedAttention (KV Cache)                            │
│  - Qwen2.5-7B-Instruct-AWQ                              │
│  GPU: NVIDIA (16GB+ VRAM)                               │
└─────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
cd vllm-deployment-study

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 HF_TOKEN 설정
```

### 2. 서비스 시작

```bash
# 전체 서비스 시작 (GPU 필요)
docker-compose up -d

# 또는 개별 시작
docker-compose up -d inference-server
docker-compose up -d api-server

# 로그 확인
docker-compose logs -f
```

### 3. API 테스트

```bash
# Health Check
curl http://localhost:8000/health

# Chat Completion
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "message": "토마토 100평 재배 시 예상 수확량을 알려주세요.",
    "max_tokens": 256,
    "temperature": 0.7
  }'

# Tool Calling
curl -X POST http://localhost:8000/api/v1/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "query": "내일 서울 날씨를 알려줘",
    "available_tools": ["get_weather", "predict_temperature"]
  }'
```

### 4. 벤치마크 실행

```bash
# Latency 테스트
python benchmarks/latency_test.py

# Throughput 테스트
python benchmarks/throughput_test.py

# GPU 사용률 모니터링
python benchmarks/gpu_utilization.py 60

# Locust 부하 테스트
locust -f benchmarks/load_test.py --host http://localhost:8000
```

## 프로젝트 구조

```
vllm-deployment-study/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
│
├── inference_server/          # GPU 추론 서버
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── vllm_server.py
│   └── config.yaml
│
├── api_server/                # 비즈니스 로직 API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── routers/
│   ├── services/
│   └── models/
│
├── benchmarks/                # 성능 벤치마크
│   ├── load_test.py
│   ├── latency_test.py
│   ├── throughput_test.py
│   ├── gpu_utilization.py
│   └── results/
│
├── notebooks/                 # 실험 노트북
│   ├── 01_vllm_basics.ipynb
│   ├── 02_performance_tuning.ipynb
│   └── 03_comparison.ipynb
│
├── docs/                      # 문서
│   ├── architecture.md
│   ├── vllm_internals.md
│   └── deployment_guide.md
│
├── scripts/                   # 유틸리티 스크립트
│   ├── start_services.sh
│   ├── stop_services.sh
│   └── run_benchmark.sh
│
└── tests/                     # 테스트
    ├── test_inference_server.py
    └── test_api_server.py
```

## 주요 설정

### vLLM 설정 (inference_server/config.yaml)

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct-AWQ"
  quantization: "awq"

engine:
  gpu_memory_utilization: 0.90
  max_num_batched_tokens: 8192
  max_num_seqs: 256
```

### Docker Compose 주요 설정

- **inference-server**: GPU 컨테이너, 포트 8001
- **api-server**: CPU 컨테이너, 포트 8000
- 내부 네트워크로 통신

## 성능 벤치마크 요약

| 메트릭 | 값 |
|--------|-----|
| Latency P50 | ~130ms |
| Latency P95 | ~270ms |
| Throughput (10 users) | ~42 req/s |
| GPU Memory Usage | ~18GB/24GB |
| GPU Utilization | ~78% |

자세한 벤치마크 결과는 `benchmarks/results/` 디렉토리를 참조하세요.

## vLLM 핵심 기술

### 1. Continuous Batching
- 동적 배치 구성으로 GPU 유휴 시간 최소화
- 완료된 요청 즉시 제거, 새 요청 즉시 추가

### 2. PagedAttention
- KV Cache를 블록 단위로 관리
- 메모리 효율 20~30% 향상

자세한 내용은 `docs/vllm_internals.md`를 참조하세요.

## 개발 가이드

### 로컬 개발 (GPU 없이)

```bash
# API 서버만 로컬 실행
cd api_server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Mock 모드로 테스트
export INFERENCE_SERVER_URL=http://localhost:8001
export MOCK_MODE=true
```

### 테스트 실행

```bash
# 단위 테스트
pytest tests/

# 통합 테스트 (서비스 실행 필요)
pytest tests/ --integration
```

## 모니터링 (선택적)

```bash
# Prometheus + Grafana 포함 시작
docker-compose --profile monitoring up -d

# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## 트러블슈팅

### CUDA OOM 에러
```bash
# GPU 메모리 사용률 낮추기
# config.yaml에서:
gpu_memory_utilization: 0.80  # 0.90 → 0.80
```

### 모델 로딩 실패
```bash
# HuggingFace 토큰 확인
echo $HF_TOKEN

# 모델 캐시 확인
ls -la ./models_cache/
```

### API 서버 연결 실패
```bash
# Inference 서버 상태 확인
curl http://localhost:8001/health

# 네트워크 확인
docker network inspect vllm-deployment-study_ai-network
```

## 라이선스

MIT License

## 참고 자료

- [vLLM 공식 문서](https://docs.vllm.ai/)
- [PagedAttention 논문](https://arxiv.org/abs/2309.06180)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
