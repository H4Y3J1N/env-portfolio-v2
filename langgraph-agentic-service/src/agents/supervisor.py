"""
Supervisor Agent

Multi-Agent 오케스트레이션을 담당하는 Supervisor Pattern 구현
"""

import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .state import AgentState, SupervisorDecision
from .rag_agent import RAGAgent
from .tool_agent import ToolAgent
from .conversation_agent import ConversationAgent

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Supervisor Pattern Multi-Agent 오케스트레이터

    사용자 요청을 분석하고 적절한 Agent를 선택하여 실행
    """

    SUPERVISOR_PROMPT = """당신은 농업 AI 시스템의 Supervisor입니다.
사용자 요청을 분석하고 적절한 Agent를 선택하세요.

## 사용 가능한 Agent
1. **rag**: 문서 검색이 필요할 때 (농업 매뉴얼, 가이드, 정보 조회 등)
2. **tool**: 외부 도구 호출이 필요할 때 (수확량 예측, 날씨 조회, 패널 최적화 등)
3. **conversation**: 일반 대화, 이전 맥락 참조, 간단한 질문 응답
4. **respond**: 충분한 정보가 모였으므로 최종 응답 생성
5. **end**: 이미 응답이 완료됨

## 현재 상태
- 사용자 메시지: {user_input}
- 검색된 문서 수: {num_docs}
- Tool 실행 결과: {tool_results}
- 반복 횟수: {iteration_count}/5
- 이전 판단: {previous_reasoning}

## 결정 기준
- 정보 검색이 필요하면 → rag
- 예측/계산이 필요하면 → tool
- 단순 대화나 맥락 정리가 필요하면 → conversation
- 답변할 준비가 되었으면 → respond
- 이미 final_response가 있으면 → end

## 응답 형식 (JSON)
{{"next_agent": "rag|tool|conversation|respond|end", "reasoning": "선택 이유"}}

JSON만 출력하세요:"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 5,
    ):
        self.llm = llm or ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )
        self.max_iterations = max_iterations
        self.parser = JsonOutputParser(pydantic_object=SupervisorDecision)
        self.prompt = ChatPromptTemplate.from_template(self.SUPERVISOR_PROMPT)
        self.chain = self.prompt | self.llm | self.parser

    def decide(self, state: AgentState) -> AgentState:
        """다음 Agent 결정"""

        # 최대 반복 횟수 체크
        if state["iteration_count"] >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            return {
                **state,
                "next_agent": "respond",
                "reasoning": "최대 반복 횟수 도달, 응답 생성으로 전환",
            }

        # 이미 최종 응답이 있으면 종료
        if state.get("final_response"):
            return {
                **state,
                "next_agent": "end",
                "reasoning": "최종 응답 완료",
            }

        try:
            # LLM으로 결정
            decision = self.chain.invoke({
                "user_input": state["user_input"],
                "num_docs": len(state.get("retrieved_docs", [])),
                "tool_results": self._format_tool_results(state.get("tool_results", [])),
                "iteration_count": state["iteration_count"],
                "previous_reasoning": state.get("reasoning", "없음"),
            })

            logger.info(f"Supervisor decision: {decision['next_agent']} - {decision['reasoning']}")

            return {
                **state,
                "next_agent": decision["next_agent"],
                "reasoning": decision["reasoning"],
                "iteration_count": state["iteration_count"] + 1,
            }

        except Exception as e:
            logger.error(f"Supervisor decision error: {e}")
            # 에러 시 응답 생성으로 전환
            return {
                **state,
                "next_agent": "respond",
                "reasoning": f"결정 오류: {str(e)}, 응답 생성으로 전환",
                "error": str(e),
            }

    def _format_tool_results(self, tool_results: list[dict]) -> str:
        """Tool 결과 포맷팅"""
        if not tool_results:
            return "없음"

        formatted = []
        for result in tool_results:
            status = "성공" if result.get("success", False) else "실패"
            formatted.append(f"- {result.get('tool_name', 'unknown')}: {status}")

        return "\n".join(formatted)


def create_agent_graph(
    supervisor: SupervisorAgent,
    rag_agent: RAGAgent,
    tool_agent: ToolAgent,
    conversation_agent: ConversationAgent,
) -> StateGraph:
    """Supervisor Pattern StateGraph 생성

    Args:
        supervisor: Supervisor Agent
        rag_agent: RAG 검색 Agent
        tool_agent: Tool 실행 Agent
        conversation_agent: 대화 관리 Agent

    Returns:
        컴파일된 StateGraph
    """

    # StateGraph 생성
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("supervisor", supervisor.decide)
    graph.add_node("rag", rag_agent.execute)
    graph.add_node("tool", tool_agent.execute)
    graph.add_node("conversation", conversation_agent.execute)
    graph.add_node("respond", conversation_agent.generate_response)

    # 조건부 라우팅 함수
    def route_by_decision(state: AgentState) -> str:
        next_agent = state.get("next_agent", "end")
        logger.debug(f"Routing to: {next_agent}")
        return next_agent

    # Supervisor에서 각 Agent로 조건부 엣지
    graph.add_conditional_edges(
        "supervisor",
        route_by_decision,
        {
            "rag": "rag",
            "tool": "tool",
            "conversation": "conversation",
            "respond": "respond",
            "end": END,
        }
    )

    # 각 Agent 실행 후 Supervisor로 복귀
    graph.add_edge("rag", "supervisor")
    graph.add_edge("tool", "supervisor")
    graph.add_edge("conversation", "supervisor")
    graph.add_edge("respond", END)

    # 시작점 설정
    graph.set_entry_point("supervisor")

    return graph.compile()


def create_simple_graph(
    rag_agent: RAGAgent,
    conversation_agent: ConversationAgent,
) -> StateGraph:
    """간단한 RAG + 응답 그래프 (Tool 없이)

    빠른 테스트용 단순화된 그래프
    """

    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("rag", rag_agent.execute)
    graph.add_node("respond", conversation_agent.generate_response)

    # 엣지: RAG → 응답 → END
    graph.add_edge("rag", "respond")
    graph.add_edge("respond", END)

    # 시작점
    graph.set_entry_point("rag")

    return graph.compile()
