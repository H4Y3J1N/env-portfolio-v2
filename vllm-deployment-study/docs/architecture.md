# vLLM Serving 아키텍처 설계

## 1. 설계 목표

### 기능 요구사항
- 높은 처리량 (Throughput)
- 낮은 지연시간 (Latency)
- GPU 자원 효율적 사용
- 독립적 스케일링 가능

### 비기능 요구사항
- 고가용성 (High Availability)
- 장애 격리 (Fault Isolation)
- 모니터링 가능성 (Observability)
- 배포 용이성 (Deployability)

---

## 2. 아키텍처 패턴

### 마이크로서비스 분리

**왜 분리하는가?**

#### 모놀리식 구조 (안 좋은 예)
```
[FastAPI + vLLM in Single Container]
           ↓
    - GPU 자원 낭비
    - API 변경 시 GPU 서버 재시작
    - 스케일링 비효율
```

#### 분리 구조 (권장)
```
[API Server] ← HTTP → [Inference Server (vLLM)]
      ↓                        ↓
  - 비즈니스 로직           - GPU 추론만
  - 인증/인가              - 메모리 최적화
  - CPU만 사용             - 고정된 설정
```

**장점**:
1. **자원 최적화**: GPU는 추론에만 집중
2. **독립 배포**: API 수정 시 추론 서버 무영향
3. **독립 스케일링**: 각각 필요에 따라 확장
4. **장애 격리**: API 에러가 GPU 서버 영향 X

---

## 3. 컴포넌트 상세

### 3.1 API Server (FastAPI)

**역할**:
- 클라이언트 요청 검증
- 프롬프트 엔지니어링
- 응답 후처리
- 인증/인가
- 로깅 및 메트릭

**기술 스택**:
- FastAPI (비동기 웹 프레임워크)
- httpx (비동기 HTTP 클라이언트)
- Pydantic (데이터 검증)

**엔드포인트**:
```
GET  /                          # 서비스 정보
GET  /health                    # 헬스 체크
GET  /ready                     # Readiness (K8s)
POST /api/v1/chat/completions   # 채팅 완성
POST /api/v1/tools/call         # Tool Calling
GET  /api/v1/tools/available    # 도구 목록
```

### 3.2 Inference Server (vLLM)

**역할**:
- 고성능 LLM 추론
- KV Cache 관리
- 동적 배치 처리

**기술 스택**:
- vLLM (추론 엔진)
- CUDA (GPU 가속)
- AWQ (4-bit 양자화)

**엔드포인트**:
```
GET  /health     # 헬스 체크
GET  /metrics    # 메트릭
POST /generate   # 텍스트 생성
POST /generate/stream  # 스트리밍 생성
```

---

## 4. 통신 흐름

### 요청 처리 흐름

```
1. Client → API Server
   - HTTP Request
   - 요청 검증

2. API Server: Prompt Engineering
   - 시스템 프롬프트 추가
   - 도구 정보 주입
   - Qwen2.5 형식으로 포맷팅

3. API Server → Inference Server
   - Internal HTTP Request
   - 비동기 처리

4. Inference Server: 추론
   - Continuous Batching
   - PagedAttention
   - 토큰 생성

5. Inference Server → API Server
   - 생성 결과 반환

6. API Server: 후처리
   - Tool Call 추출
   - 응답 포맷팅

7. API Server → Client
   - HTTP Response
```

---

## 5. 설계 트레이드오프

### 5.1 GPU Memory Utilization

| 설정 | 장점 | 단점 | 권장 상황 |
|------|------|------|----------|
| 0.70 | 안정적, OOM 거의 없음 | 처리량 낮음 | 개발/테스트 |
| 0.90 | 높은 처리량 | 간헐적 OOM 가능 | **프로덕션 권장** |
| 0.95 | 최대 처리량 | OOM 위험 높음 | 피크 타임 전용 |

### 5.2 Max Num Sequences

| 설정 | 처리량 | 지연시간 | GPU 메모리 |
|------|--------|---------|-----------|
| 64   | 낮음   | 매우 낮음 | 적음 |
| 128  | 중간   | 낮음    | 중간 |
| **256** | **높음** | **중간** | **높음 (권장)** |
| 512  | 최대   | 높음    | 매우 높음 |

### 5.3 Quantization

| 방식 | 정확도 | 속도 | 메모리 |
|------|--------|------|--------|
| FP16 | 100% | 1x | 100% |
| **AWQ** | **99%** | **1.8x** | **25%** |
| GPTQ | 98% | 1.5x | 25% |
| INT8 | 95% | 2.2x | 50% |

---

## 6. 프로덕션 고려사항

### 6.1 Health Check

```python
# Inference Server
@app.get("/health")
async def health():
    if engine is None:
        return {"status": "initializing"}
    return {"status": "healthy"}

# API Server
@app.get("/health")
async def health():
    inference_health = await check_inference_server()
    return {
        "api": "healthy",
        "inference": inference_health
    }
```

### 6.2 Graceful Shutdown

```python
@app.on_event("shutdown")
async def shutdown():
    # 진행 중인 요청 완료 대기
    await engine.wait_for_completion()
    # 리소스 정리
    await engine.cleanup()
```

### 6.3 Error Handling

```python
try:
    result = await inference_client.generate(...)
except TimeoutException:
    return fallback_response()
except InferenceServerError:
    logger.error("Inference failed")
    raise HTTPException(503)
```

---

## 7. 스케일링 전략

### 7.1 Horizontal Scaling (추론 서버)

```
        [Load Balancer]
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
[vLLM-1]  [vLLM-2]  [vLLM-3]
(GPU 0)   (GPU 1)   (GPU 2)
```

### 7.2 Vertical Scaling (API 서버)

```
[API-1] [API-2] [API-3] [API-4]
   ↓       ↓       ↓       ↓
      [Shared Inference Server]
```

---

## 8. 보안 고려사항

### 네트워크 격리
- Inference 서버: 내부 네트워크만 노출
- API 서버: Public 접근 가능

### 인증/인가
- API 서버에서만 구현
- Inference 서버는 내부망으로 보호

### Rate Limiting
- API 서버에서 구현
- IP/API Key 기반 제한

---

## 9. 모니터링

### 핵심 메트릭

**Inference Server**:
- requests_per_second
- tokens_per_second
- gpu_utilization
- queue_length

**API Server**:
- request_latency
- error_rate
- active_connections

### 알림 (Alerting)

```yaml
groups:
  - name: vllm_alerts
    rules:
      - alert: HighLatency
        expr: vllm_request_latency_p95 > 500
        for: 5m

      - alert: HighErrorRate
        expr: api_error_rate > 0.05
        for: 2m
```
