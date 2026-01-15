"""
FastAPI Application

메인 애플리케이션 진입점
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat_router, agent_router, health_router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    logger.info("Starting LangGraph Agentic Service...")

    # 서비스 초기화
    try:
        from ..services import initialize_services
        await initialize_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.warning(f"Service initialization warning: {e}")

    yield

    # Shutdown
    logger.info("Shutting down LangGraph Agentic Service...")


def create_app() -> FastAPI:
    """FastAPI 앱 생성"""

    app = FastAPI(
        title="LangGraph Agentic Service",
        description="""
        농업 도메인 특화 Multi-Agent RAG 서비스

        ## 주요 기능
        - **Multi-Agent**: LangGraph Supervisor Pattern 기반 오케스트레이션
        - **Hybrid RAG**: BGE-M3 (Dense + Sparse) + Cross-encoder Reranker
        - **SSE Streaming**: 실시간 토큰 스트리밍
        - **Session Management**: Redis 기반 Multi-turn 대화 관리
        - **Monitoring**: Langfuse 토큰 사용량 추적

        ## 엔드포인트
        - `POST /api/v1/chat/stream`: SSE 스트리밍 채팅
        - `POST /api/v1/chat/completions`: 일반 채팅
        - `POST /api/v1/agent/execute`: Multi-Agent 실행
        """,
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")

    return app


# 앱 인스턴스
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
