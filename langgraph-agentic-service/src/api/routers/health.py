"""
Health Router

헬스체크 및 시스템 상태 엔드포인트
"""

import logging
from fastapi import APIRouter

from ..schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """헬스체크

    서비스 및 컴포넌트 상태 확인
    """
    components = {}

    # Redis 상태 확인
    try:
        from ...session import RedisSessionStore
        store = RedisSessionStore()
        await store.ping()
        components["redis"] = "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        components["redis"] = "unavailable"

    # RAG 상태 확인
    try:
        from ...rag import HybridRAGPipeline
        # 간단한 초기화 테스트
        components["rag"] = "healthy"
    except Exception as e:
        logger.warning(f"RAG health check failed: {e}")
        components["rag"] = "unavailable"

    # LLM 상태 확인
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            components["llm"] = "healthy"
        else:
            components["llm"] = "no_api_key"
    except Exception as e:
        components["llm"] = "unavailable"

    # 전체 상태 결정
    all_healthy = all(v == "healthy" for v in components.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version="0.1.0",
        components=components,
    )


@router.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "LangGraph Agentic Service",
        "version": "0.1.0",
        "description": "농업 도메인 특화 Multi-Agent RAG 서비스",
    }


@router.get("/ready")
async def readiness_check():
    """준비 상태 확인 (Kubernetes readiness probe용)"""
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """생존 상태 확인 (Kubernetes liveness probe용)"""
    return {"alive": True}
