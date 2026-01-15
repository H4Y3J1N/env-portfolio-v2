# ===========================================
# vLLM Inference Server Dockerfile
# ===========================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# pip 업그레이드
RUN pip3 install --no-cache-dir --upgrade pip

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 서버 코드 복사
COPY vllm_server.py .
COPY config.yaml .

# 포트 노출
EXPOSE 8001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 서버 실행
CMD ["python3", "vllm_server.py"]
