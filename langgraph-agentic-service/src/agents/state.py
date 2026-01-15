"""
LangGraph State Definition

Multi-Agent 시스템의 공유 상태 정의
"""

from typing import TypedDict, Annotated, Sequence, Literal, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Document(BaseModel):
    """검색된 문서"""
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float = 0.0


class ToolResult(BaseModel):
    """Tool 실행 결과"""
    tool_name: str
    arguments: dict
    result: Any
    success: bool = True
    error: Optional[str] = None


class AgentState(TypedDict):
    """Multi-Agent 공유 상태

    LangGraph StateGraph에서 모든 노드가 공유하는 상태
    """

    # 대화 메시지 히스토리 (LangGraph add_messages 리듀서 사용)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 현재 사용자 입력
    user_input: str

    # 다음에 실행할 Agent
    next_agent: Literal["rag", "tool", "conversation", "respond", "end"]

    # Supervisor의 판단 이유
    reasoning: str

    # RAG 검색 결과
    retrieved_docs: list[dict]

    # Tool 실행 결과
    tool_results: list[dict]

    # 최종 응답
    final_response: str

    # 세션 정보
    session_id: str
    tenant_id: str

    # 반복 카운터 (무한 루프 방지)
    iteration_count: int

    # 에러 정보
    error: Optional[str]


class SupervisorDecision(BaseModel):
    """Supervisor Agent의 결정"""

    next_agent: Literal["rag", "tool", "conversation", "respond", "end"] = Field(
        description="다음에 실행할 Agent"
    )
    reasoning: str = Field(
        description="해당 Agent를 선택한 이유"
    )


def create_initial_state(
    user_input: str,
    session_id: str = "default",
    tenant_id: str = "default"
) -> AgentState:
    """초기 상태 생성"""
    return AgentState(
        messages=[],
        user_input=user_input,
        next_agent="rag",  # 기본적으로 RAG부터 시작
        reasoning="",
        retrieved_docs=[],
        tool_results=[],
        final_response="",
        session_id=session_id,
        tenant_id=tenant_id,
        iteration_count=0,
        error=None,
    )
