# 배포 가이드

## 1. 사전 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (16GB+ VRAM 권장)
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ (모델 캐시)

### 소프트웨어
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- CUDA 12.1+ Driver

---

## 2. 로컬 개발 환경

### 2.1 저장소 클론

```bash
git clone <repository-url>
cd vllm-deployment-study
```

### 2.2 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# .env 파일 수정
# HF_TOKEN=hf_your_token_here
```

### 2.3 서비스 시작

```bash
# 전체 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 상태 확인
docker-compose ps
```

### 2.4 테스트

```bash
# Health Check
curl http://localhost:8000/health

# API 테스트
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요!", "max_tokens": 100}'
```

---

## 3. 프로덕션 배포

### 3.1 Docker Image 빌드

```bash
# Inference Server
docker build -t vllm-inference:latest ./inference_server

# API Server
docker build -t vllm-api:latest ./api_server
```

### 3.2 환경별 설정

#### 개발 환경
```yaml
# docker-compose.override.yml
services:
  inference-server:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./inference_server:/app:ro
```

#### 프로덕션 환경
```yaml
# docker-compose.prod.yml
services:
  inference-server:
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "3"
```

### 3.3 프로덕션 시작

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 4. Kubernetes 배포

### 4.1 Namespace 생성

```bash
kubectl create namespace vllm-serving
```

### 4.2 Secret 생성

```bash
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN=hf_your_token_here \
  -n vllm-serving
```

### 4.3 Deployment 예시

```yaml
# kubernetes/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: vllm-serving
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      containers:
      - name: vllm
        image: vllm-inference:latest
        ports:
        - containerPort: 8001
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: HF_TOKEN
        resources:
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 180
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 180
          periodSeconds: 10
```

### 4.4 Service 예시

```yaml
# kubernetes/inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-inference
  namespace: vllm-serving
spec:
  selector:
    app: vllm-inference
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
```

### 4.5 배포

```bash
kubectl apply -f kubernetes/
```

---

## 5. 모니터링 설정

### 5.1 Prometheus 스택

```bash
# 모니터링 포함 시작
docker-compose --profile monitoring up -d
```

### 5.2 Grafana 대시보드

1. Grafana 접속: http://localhost:3000
2. 기본 계정: admin / admin
3. Data Source 추가: Prometheus (http://prometheus:9090)
4. 대시보드 임포트

### 5.3 알림 설정

```yaml
# monitoring/prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/alert.rules.yml
```

---

## 6. 스케일링

### 6.1 수평 스케일링 (API Server)

```bash
# Docker Compose
docker-compose up -d --scale api-server=3
```

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 6.2 Multi-GPU (Inference Server)

```yaml
# inference_server/config.yaml
engine:
  tensor_parallel_size: 2  # 2 GPU 사용
```

---

## 7. 트러블슈팅

### 7.1 일반적인 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| 컨테이너 시작 실패 | GPU 드라이버 문제 | `nvidia-smi` 확인 |
| 모델 로딩 실패 | HF_TOKEN 누락 | 환경 변수 확인 |
| OOM 에러 | 메모리 부족 | gpu_memory_utilization 낮추기 |
| 연결 거부 | 서비스 미시작 | Health Check 대기 |

### 7.2 로그 확인

```bash
# Docker 로그
docker-compose logs -f inference-server
docker-compose logs -f api-server

# Kubernetes 로그
kubectl logs -f deployment/vllm-inference -n vllm-serving
```

### 7.3 디버그 모드

```bash
# 환경 변수로 디버그 모드 활성화
LOG_LEVEL=DEBUG docker-compose up
```

---

## 8. 보안 체크리스트

- [ ] HF_TOKEN을 Secret으로 관리
- [ ] Inference Server는 내부 네트워크만 노출
- [ ] API Server에 Rate Limiting 적용
- [ ] HTTPS 적용 (Ingress/Load Balancer)
- [ ] 로그에 민감 정보 미포함 확인
- [ ] 컨테이너 비root 사용자 실행

---

## 9. 백업 및 복구

### 모델 캐시 백업

```bash
# 백업
tar -czvf models_cache_backup.tar.gz ./models_cache

# 복구
tar -xzvf models_cache_backup.tar.gz
```

### 설정 백업

```bash
# 백업
cp -r configs/ configs_backup/
cp .env .env.backup
```
