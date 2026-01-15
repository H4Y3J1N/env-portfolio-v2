"""
vLLM Inference Server

고성능 LLM 추론 서버 (Continuous Batching + PagedAttention)
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# vLLM imports
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Running in mock mode.")

# ===========================================
# Logging Setup
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================
# Configuration
# ===========================================
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# ===========================================
# Global Variables
# ===========================================
engine: Optional[AsyncLLMEngine] = None
startup_time: float = 0
request_count: int = 0

# ===========================================
# Pydantic Models
# ===========================================
class GenerateRequest(BaseModel):
    """추론 요청 스키마"""
    prompt: str = Field(..., description="입력 프롬프트")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="최대 생성 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="샘플링 온도")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 샘플링")
    top_k: int = Field(default=50, ge=0, description="Top-k 샘플링")
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="반복 패널티")
    stop: Optional[List[str]] = Field(default=None, description="중단 토큰 목록")
    stream: bool = Field(default=False, description="스트리밍 모드")


class GenerateResponse(BaseModel):
    """추론 응답 스키마"""
    text: str = Field(..., description="생성된 텍스트")
    tokens_used: int = Field(..., description="사용된 토큰 수")
    finish_reason: str = Field(..., description="종료 이유")
    latency_ms: float = Field(..., description="처리 시간 (ms)")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model: Optional[str] = None
    quantization: Optional[str] = None
    uptime_seconds: Optional[float] = None
    total_requests: Optional[int] = None


class MetricsResponse(BaseModel):
    """메트릭 응답"""
    model: str
    gpu_memory_utilization: float
    max_num_seqs: int
    uptime_seconds: float
    total_requests: int


# ===========================================
# Lifespan (Startup/Shutdown)
# ===========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    global engine, startup_time

    logger.info("=" * 60)
    logger.info("vLLM Inference Server Starting...")
    logger.info("=" * 60)

    if VLLM_AVAILABLE:
        try:
            model_config = config['model']
            engine_config = config['engine']

            logger.info(f"Loading model: {model_config['name']}")
            logger.info(f"Quantization: {model_config.get('quantization', 'none')}")
            logger.info(f"GPU Memory Utilization: {engine_config['gpu_memory_utilization']}")

            # AsyncEngineArgs 설정
            engine_args = AsyncEngineArgs(
                model=model_config['name'],
                tokenizer=model_config.get('tokenizer'),
                quantization=model_config.get('quantization'),
                dtype=model_config.get('dtype', 'auto'),
                gpu_memory_utilization=engine_config['gpu_memory_utilization'],
                max_num_batched_tokens=engine_config.get('max_num_batched_tokens', 8192),
                max_num_seqs=engine_config.get('max_num_seqs', 256),
                max_model_len=engine_config.get('max_model_len'),
                block_size=engine_config.get('block_size', 16),
                enable_prefix_caching=engine_config.get('enable_prefix_caching', False),
                disable_log_stats=engine_config.get('disable_log_stats', False),
                trust_remote_code=model_config.get('trust_remote_code', True),
            )

            # 엔진 초기화
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            startup_time = time.time()

            logger.info("=" * 60)
            logger.info("vLLM Engine initialized successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    else:
        logger.warning("Running in MOCK mode (vLLM not available)")
        startup_time = time.time()

    yield

    # Shutdown
    logger.info("Shutting down vLLM engine...")
    if engine is not None:
        # Cleanup if needed
        pass
    logger.info("vLLM Inference Server stopped.")


# ===========================================
# FastAPI App
# ===========================================
app = FastAPI(
    title="vLLM Inference Server",
    description="High-performance LLM inference with Continuous Batching and PagedAttention",
    version="1.0.0",
    lifespan=lifespan
)


# ===========================================
# Endpoints
# ===========================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    if engine is None and VLLM_AVAILABLE:
        return HealthResponse(status="initializing")

    uptime = time.time() - startup_time if startup_time > 0 else 0

    return HealthResponse(
        status="healthy",
        model=config['model']['name'],
        quantization=config['model'].get('quantization'),
        uptime_seconds=round(uptime, 2),
        total_requests=request_count
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """메트릭 엔드포인트"""
    if engine is None and VLLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    uptime = time.time() - startup_time if startup_time > 0 else 0

    return MetricsResponse(
        model=config['model']['name'],
        gpu_memory_utilization=config['engine']['gpu_memory_utilization'],
        max_num_seqs=config['engine'].get('max_num_seqs', 256),
        uptime_seconds=round(uptime, 2),
        total_requests=request_count
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """텍스트 생성 엔드포인트"""
    global request_count

    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Generate request received")
    logger.debug(f"[{request_id}] Prompt length: {len(request.prompt)} chars")

    # Mock 모드
    if not VLLM_AVAILABLE or engine is None:
        await asyncio.sleep(0.1)  # Simulate processing
        request_count += 1
        latency = (time.time() - start_time) * 1000

        return GenerateResponse(
            text="[MOCK] This is a mock response. vLLM is not available.",
            tokens_used=10,
            finish_reason="mock",
            latency_ms=round(latency, 2)
        )

    # Sampling Parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        stop=request.stop
    )

    try:
        # vLLM 추론 (비동기)
        results_generator = engine.generate(
            request.prompt,
            sampling_params,
            request_id
        )

        # 결과 수집
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")

        # 응답 생성
        output = final_output.outputs[0]
        request_count += 1
        latency = (time.time() - start_time) * 1000

        logger.info(f"[{request_id}] Generation complete: {len(output.token_ids)} tokens, {latency:.2f}ms")

        return GenerateResponse(
            text=output.text,
            tokens_used=len(output.token_ids),
            finish_reason=output.finish_reason or "unknown",
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"[{request_id}] Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """스트리밍 텍스트 생성 엔드포인트"""

    if not VLLM_AVAILABLE or engine is None:
        raise HTTPException(status_code=503, detail="Streaming not available in mock mode")

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Stream request received")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        stop=request.stop
    )

    async def generate_tokens() -> AsyncGenerator[str, None]:
        """토큰 스트리밍 제너레이터"""
        try:
            results_generator = engine.generate(
                request.prompt,
                sampling_params,
                request_id
            )

            previous_text = ""
            async for request_output in results_generator:
                output = request_output.outputs[0]
                new_text = output.text[len(previous_text):]
                previous_text = output.text

                if new_text:
                    yield f"data: {new_text}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream"
    )


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "vLLM Inference Server",
        "version": "1.0.0",
        "model": config['model']['name'],
        "docs": "/docs"
    }


# ===========================================
# Main
# ===========================================
if __name__ == "__main__":
    server_config = config.get('server', {})

    uvicorn.run(
        app,
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8001),
        log_level="info",
        timeout_keep_alive=server_config.get('timeout', 600)
    )
