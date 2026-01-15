"""
API Request/Response Schemas

Pydantic 모델 정의
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """채팅 요청"""

    message: str = Field(..., description="사용자 메시지", min_length=1)
    session_id: Optional[str] = Field(None, description="세션 ID (없으면 자동 생성)")
    tenant_id: str = Field("default", description="테넌트 ID")
    stream: bool = Field(True, description="스트리밍 응답 여부")
    max_tokens: int = Field(1024, description="최대 토큰 수", ge=1, le=4096)
    temperature: float = Field(0.7, description="생성 temperature", ge=0, le=2)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "토마토 수확량을 예측해줘",
                    "session_id": "session-123",
                    "stream": True,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """채팅 응답 (non-streaming)"""

    response: str = Field(..., description="AI 응답")
    session_id: str = Field(..., description="세션 ID")
    retrieved_docs: List[Dict] = Field(default_factory=list, description="검색된 문서")
    tool_results: List[Dict] = Field(default_factory=list, description="Tool 실행 결과")
    tokens_used: Optional[int] = Field(None, description="사용된 토큰 수")


class AgentRequest(BaseModel):
    """Agent 실행 요청"""

    query: str = Field(..., description="사용자 쿼리")
    session_id: Optional[str] = Field(None, description="세션 ID")
    tenant_id: str = Field("default", description="테넌트 ID")
    use_rag: bool = Field(True, description="RAG 사용 여부")
    use_tools: bool = Field(True, description="Tool 사용 여부")
    max_iterations: int = Field(5, description="최대 반복 횟수", ge=1, le=10)


class AgentResponse(BaseModel):
    """Agent 실행 응답"""

    response: str = Field(..., description="최종 응답")
    session_id: str = Field(..., description="세션 ID")
    iterations: int = Field(..., description="실행된 반복 횟수")
    agents_used: List[str] = Field(default_factory=list, description="사용된 Agent 목록")
    retrieved_docs: List[Dict] = Field(default_factory=list, description="검색된 문서")
    tool_results: List[Dict] = Field(default_factory=list, description="Tool 실행 결과")
    reasoning_trace: List[str] = Field(default_factory=list, description="Supervisor 판단 기록")


class SessionResponse(BaseModel):
    """세션 정보 응답"""

    session_id: str = Field(..., description="세션 ID")
    tenant_id: str = Field(..., description="테넌트 ID")
    message_count: int = Field(..., description="메시지 수")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="마지막 업데이트 시간")
    messages: List[Dict] = Field(default_factory=list, description="대화 히스토리")


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    components: Dict[str, str] = Field(default_factory=dict, description="컴포넌트별 상태")


class SSEEvent(BaseModel):
    """SSE 이벤트"""

    event: str = Field(..., description="이벤트 타입")
    data: Dict[str, Any] = Field(default_factory=dict, description="이벤트 데이터")

    def to_sse_format(self) -> str:
        """SSE 포맷 문자열 반환"""
        import json
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"
