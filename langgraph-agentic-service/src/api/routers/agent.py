"""
Agent Router

Multi-Agent 실행 엔드포인트
"""

import logging
import uuid
from fastapi import APIRouter, HTTPException

from ..schemas import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest) -> AgentResponse:
    """Multi-Agent 실행

    Supervisor Pattern으로 여러 Agent를 오케스트레이션

    - **query**: 사용자 쿼리
    - **use_rag**: RAG 사용 여부
    - **use_tools**: Tool 사용 여부
    - **max_iterations**: 최대 반복 횟수
    """
    from ...services import AgentService

    logger.info(f"Agent execution request: {request.query[:50]}...")

    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Agent 서비스 초기화
        agent_service = AgentService(
            use_rag=request.use_rag,
            use_tools=request.use_tools,
            max_iterations=request.max_iterations,
            use_mock=True,  # TODO: 실제 서비스 사용
        )

        # Agent 실행
        result = await agent_service.execute(
            message=request.query,
            session_id=session_id,
            tenant_id=request.tenant_id,
        )

        return AgentResponse(
            response=result.get("response", ""),
            session_id=session_id,
            iterations=result.get("iterations", 0),
            agents_used=result.get("agents_used", []),
            retrieved_docs=result.get("retrieved_docs", []),
            tool_results=result.get("tool_results", []),
            reasoning_trace=result.get("reasoning_trace", []),
        )

    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}")
async def get_agent_status(session_id: str):
    """Agent 실행 상태 조회

    현재 진행 중인 Agent 실행 상태를 조회
    """
    # TODO: 실제 상태 조회 구현
    return {
        "session_id": session_id,
        "status": "idle",
        "current_agent": None,
        "iteration": 0,
    }
