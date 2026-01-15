"""
Agriculture AI API Server

FastAPI 기반 비즈니스 로직 API 서버
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import chat, tool_calling
from services.inference_client import InferenceClient

# ===========================================
# Logging Setup
# ===========================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================
# Global Variables
# ===========================================
startup_time: float = 0


# ===========================================
# Lifespan
# ===========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    global startup_time

    logger.info("=" * 60)
    logger.info("Agriculture AI API Server Starting...")
    logger.info("=" * 60)

    startup_time = time.time()

    # Inference 서버 연결 확인
    client = InferenceClient()
    try:
        health = await client.health_check()
        logger.info(f"Inference server connected: {health}")
    except Exception as e:
        logger.warning(f"Inference server not available: {e}")
        logger.warning("API server will start but inference may fail")

    logger.info("API Server ready!")
    logger.info("=" * 60)

    yield

    logger.info("API Server shutting down...")


# ===========================================
# FastAPI App
# ===========================================
app = FastAPI(
    title="Agriculture AI API",
    description="""
    농업 도메인 특화 AI API 서버

    ## 기능
    - **Chat Completion**: 일반 대화 응답
    - **Tool Calling**: 농업 도구 호출 (수확량 예측, 기온 예측, 패널 각도 최적화 등)

    ## 아키텍처
    - API Server (FastAPI) → Inference Server (vLLM)
    - 마이크로서비스 분리로 독립적 스케일링
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ===========================================
# Middleware
# ===========================================
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()

    # 요청 로깅
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    # 응답 시간 계산
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} ({process_time:.2f}ms)")

    # 응답 헤더에 처리 시간 추가
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    return response


# ===========================================
# Exception Handlers
# ===========================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 핸들러"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error"
        }
    )


# ===========================================
# Routers
# ===========================================
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(tool_calling.router, prefix="/api/v1/tools", tags=["Tool Calling"])


# ===========================================
# Health & Status Endpoints
# ===========================================
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Agriculture AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    # Inference 서버 연결 확인
    client = InferenceClient()

    try:
        inference_health = await client.health_check()
        inference_status = "healthy"
    except Exception as e:
        logger.warning(f"Inference server health check failed: {e}")
        inference_health = {"error": str(e)}
        inference_status = "unhealthy"

    uptime = time.time() - startup_time if startup_time > 0 else 0

    return {
        "api_server": {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2)
        },
        "inference_server": {
            "status": inference_status,
            "details": inference_health
        }
    }


@app.get("/ready")
async def readiness_check():
    """Readiness 체크 (Kubernetes용)"""
    client = InferenceClient()

    try:
        await client.health_check()
        return {"ready": True}
    except Exception:
        raise HTTPException(status_code=503, detail="Inference server not ready")


# ===========================================
# Main
# ===========================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level="info"
    )
