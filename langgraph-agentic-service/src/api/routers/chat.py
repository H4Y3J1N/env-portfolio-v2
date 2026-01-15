"""
Chat Router

SSE 스트리밍 채팅 엔드포인트
"""

import logging
import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def get_agent_service():
    """Agent 서비스 의존성 주입"""
    from ...services import AgentService
    return AgentService()


async def get_session_service():
    """Session 서비스 의존성 주입"""
    from ...session import RedisSessionStore
    return RedisSessionStore()


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """SSE 기반 스트리밍 채팅

    실시간으로 토큰, RAG 결과, Tool 호출 등을 스트리밍

    - **message**: 사용자 메시지
    - **session_id**: 세션 ID (선택, 없으면 자동 생성)
    - **stream**: True면 SSE 스트리밍, False면 일반 JSON 응답
    """
    logger.info(f"Chat stream request: {request.message[:50]}...")

    if not request.stream:
        # Non-streaming 응답
        return await chat_completion(request)

    # SSE 스트리밍 응답
    return EventSourceResponse(
        chat_stream_generator(request),
        media_type="text/event-stream",
    )


async def chat_stream_generator(request: ChatRequest) -> AsyncGenerator[dict, None]:
    """SSE 이벤트 제너레이터

    이벤트 타입:
    - session: 세션 ID 정보
    - agent_start: Agent 시작
    - rag_result: RAG 검색 결과
    - tool_call: Tool 호출 정보
    - token: 생성된 토큰
    - done: 완료
    - error: 에러
    """
    from ...services import AgentService

    try:
        # 1. 세션 ID 확인/생성
        session_id = request.session_id or str(uuid.uuid4())

        yield {
            "event": "session",
            "data": json.dumps({"session_id": session_id}),
        }

        # 2. Agent 서비스 초기화
        agent_service = AgentService(use_mock=True)  # TODO: 실제 서비스 사용

        # 3. Agent 실행 (스트리밍)
        async for event in agent_service.execute_stream(
            message=request.message,
            session_id=session_id,
            tenant_id=request.tenant_id,
        ):
            event_type = event.get("type", "unknown")

            if event_type == "agent_start":
                yield {
                    "event": "agent_start",
                    "data": json.dumps({
                        "agent": event.get("agent", "unknown"),
                        "reasoning": event.get("reasoning", ""),
                    }),
                }

            elif event_type == "rag_result":
                docs = event.get("documents", [])
                yield {
                    "event": "rag_result",
                    "data": json.dumps({
                        "count": len(docs),
                        "documents": [
                            {
                                "content": doc.get("content", "")[:200],
                                "score": doc.get("score", 0),
                                "source": doc.get("metadata", {}).get("source", "unknown"),
                            }
                            for doc in docs[:3]  # 상위 3개만
                        ],
                    }),
                }

            elif event_type == "tool_call":
                yield {
                    "event": "tool_call",
                    "data": json.dumps({
                        "tool": event.get("tool", "unknown"),
                        "arguments": event.get("arguments", {}),
                        "result": event.get("result"),
                        "success": event.get("success", True),
                    }),
                }

            elif event_type == "token":
                yield {
                    "event": "token",
                    "data": json.dumps({"token": event.get("token", "")}),
                }

            elif event_type == "response":
                yield {
                    "event": "response",
                    "data": json.dumps({
                        "content": event.get("content", ""),
                        "complete": True,
                    }),
                }

        # 4. 완료
        yield {
            "event": "done",
            "data": json.dumps({"status": "completed", "session_id": session_id}),
        }

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """일반 채팅 (non-streaming)

    전체 응답을 한 번에 반환
    """
    from ...services import AgentService

    logger.info(f"Chat completion request: {request.message[:50]}...")

    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Agent 실행
        agent_service = AgentService(use_mock=True)
        result = await agent_service.execute(
            message=request.message,
            session_id=session_id,
            tenant_id=request.tenant_id,
        )

        return ChatResponse(
            response=result.get("response", ""),
            session_id=session_id,
            retrieved_docs=result.get("retrieved_docs", []),
            tool_results=result.get("tool_results", []),
            tokens_used=result.get("tokens_used"),
        )

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
