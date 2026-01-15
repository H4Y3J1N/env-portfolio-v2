# vLLM 내부 동작 원리

## 1. 전체 흐름

```
[Client Request]
       ↓
[API Server: Prompt Engineering]
       ↓
[vLLM Server: Request Queue]
       ↓
[Scheduler: Continuous Batching]
       ↓
[Model Execution: PagedAttention]
       ↓
[Token Generation]
       ↓
[Response Stream/Complete]
       ↓
[API Server: Post-processing]
       ↓
[Client Response]
```

---

## 2. Continuous Batching

### 2.1 기본 개념

**기존 방식 (Static Batching)**:
- 배치 단위로 요청 처리
- 모든 시퀀스가 끝날 때까지 대기
- GPU 유휴 시간 발생

**vLLM (Continuous Batching)**:
- Iteration 단위로 동적 배치 구성
- 완료된 시퀀스는 즉시 제거
- 새 요청 즉시 추가

### 2.2 동작 예시

**Iteration 1**:
```
Batch = [Req1(100 tokens), Req2(50 tokens), Req3(200 tokens)]
GPU 실행: 3개 시퀀스 동시 처리
```

**Iteration 2**:
```
Req2 완료! → 배치에서 제거
Req4 도착! → 배치에 추가
Batch = [Req1, Req3, Req4]
```

### 2.3 Scheduler 로직

```python
class Scheduler:
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = []
        self.max_batch_size = 256

    def schedule(self):
        # 1. 완료된 시퀀스 제거
        self.running_batch = [
            seq for seq in self.running_batch
            if not seq.is_finished()
        ]

        # 2. 대기 중인 요청 추가
        while len(self.running_batch) < self.max_batch_size:
            if not self.waiting_queue:
                break
            new_req = self.waiting_queue.pop(0)
            self.running_batch.append(new_req)

        return self.running_batch
```

### 2.4 효과

- 처리량 **2~3배** 향상
- GPU 유휴 시간 최소화
- 평균 대기 시간 감소

---

## 3. PagedAttention

### 3.1 KV Cache 문제

**Transformer의 KV Cache**:
- 각 토큰의 Key, Value를 캐싱
- 메모리: `O(batch_size × seq_len × hidden_dim)`

**문제**:
- 시퀀스 길이를 미리 알 수 없음
- 최대 길이로 할당 → 메모리 낭비

### 3.2 PagedAttention 해결책

**핵심 아이디어**:
- KV Cache를 고정 크기 블록으로 나눔
- 블록 단위로 동적 할당

**예시**:
```
Block Size = 16 tokens

Sequence 1 (100 tokens):
  [Block 0: 16 tokens]
  [Block 1: 16 tokens]
  ...
  [Block 6: 4 tokens]  ← 마지막 블록만 부분 사용

Sequence 2 (20 tokens):
  [Block 7: 16 tokens]
  [Block 8: 4 tokens]
```

### 3.3 구현 의사코드

```python
class PagedKVCache:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.blocks = []
        self.free_blocks = []

    def allocate_blocks(self, num_tokens):
        """필요한 블록 수만큼 할당"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        allocated = []
        for _ in range(num_blocks):
            if self.free_blocks:
                block = self.free_blocks.pop()
            else:
                block = self._create_new_block()
            allocated.append(block)

        return allocated

    def free_blocks(self, blocks):
        """블록 해제 (재사용 풀에 반환)"""
        self.free_blocks.extend(blocks)
```

### 3.4 효과

- 메모리 효율 **20~30%** 향상
- 동시 처리 시퀀스 수 증가
- OOM 에러 감소

---

## 4. 성능 최적화 기법

### 4.1 Prefill vs Decode

**Prefill** (초기 처리):
- 입력 프롬프트 전체를 한 번에 처리
- 병렬화 용이
- 빠름

**Decode** (토큰 생성):
- 한 번에 1개 토큰 생성
- 순차적
- 느림

**최적화**:
- Prefill은 큰 배치로
- Decode는 Continuous Batching으로

### 4.2 Speculative Decoding (선택적)

**개념**:
- 작은 모델로 여러 토큰 예측
- 큰 모델로 검증
- 맞으면 accept, 틀리면 reject

**효과**:
- Decode 속도 **2~3배** 향상

### 4.3 Tensor Parallelism (Multi-GPU)

```
    [Input]
       ↓
  ┌────┴────┐
  ↓         ↓
[GPU 0]   [GPU 1]
  ↓         ↓
  └────┬────┘
       ↓
   [Output]
```

**구현**:
```bash
vllm_server.py --tensor-parallel-size 2
```

---

## 5. 메모리 관리

### 5.1 메모리 레이아웃

```
GPU Memory (24GB)
├── Model Weights (6GB) ← AWQ 4-bit
├── KV Cache (14GB)     ← PagedAttention
├── Activation (2GB)    ← 연산 중간 결과
└── Reserved (2GB)      ← CUDA overhead
```

### 5.2 OOM 방지 전략

1. **GPU Memory Utilization 조정**:
   ```yaml
   gpu_memory_utilization: 0.90  # 90% 사용
   ```

2. **Max Sequences 제한**:
   ```yaml
   max_num_seqs: 128  # 동시 처리 제한
   ```

3. **Swap Space (선택적)**:
   - CPU RAM으로 KV Cache 스왑
   - 성능 저하 but OOM 방지

---

## 6. 실전 팁

### 6.1 최적 설정 찾기

**단계**:
1. GPU Memory Utilization = 0.70 (안전)
2. 부하 테스트 실행
3. OOM 없으면 0.80, 0.90 순차 증가
4. OOM 발생 직전 값 - 0.05 = 최적값

### 6.2 병목 지점 파악

```python
# vLLM 로그 활성화
--disable-log-stats false

# 로그 확인
# - GPU Utilization 낮음 → CPU 병목
# - Queue Length 높음 → 처리량 부족
# - Latency P99 높음 → 배치 크기 조정 필요
```

### 6.3 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| Latency 증가 | 긴 시퀀스가 배치 점유 | `max_model_len` 제한 |
| Throughput 낮음 | Batch Size 너무 작음 | `max_num_seqs` 증가 |
| OOM 발생 | Memory Utilization 높음 | 0.85로 낮추기 |
