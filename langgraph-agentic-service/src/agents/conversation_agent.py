"""
Conversation Agent

대화 관리 및 최종 응답 생성 Agent
"""

import logging
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import AgentState

logger = logging.getLogger(__name__)


class ConversationAgent:
    """대화 관리 Agent

    Multi-turn 대화 관리 및 최종 응답 생성
    """

    SYSTEM_PROMPT = """당신은 농업 도메인 전문 AI 어시스턴트입니다.
사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.

응답 시 다음 사항을 지켜주세요:
1. 검색된 문서 내용을 근거로 답변하세요
2. Tool 실행 결과가 있다면 정확한 수치를 포함하세요
3. 확실하지 않은 내용은 솔직히 밝히세요
4. 한국어로 자연스럽게 답변하세요"""

    RESPONSE_PROMPT = """## 시스템
{system_prompt}

## 사용자 질문
{user_input}

## 검색된 관련 문서
{context}

## Tool 실행 결과
{tool_results}

## 지시사항
위 정보를 종합하여 사용자 질문에 대한 답변을 생성하세요.
답변은 명확하고 구체적이어야 합니다."""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            llm: LLM 인스턴스
            model_name: 사용할 모델 이름
            temperature: 생성 temperature
            system_prompt: 시스템 프롬프트
        """
        self.llm = llm or ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.response_prompt = ChatPromptTemplate.from_template(self.RESPONSE_PROMPT)

    def execute(self, state: AgentState) -> AgentState:
        """대화 맥락 처리

        Args:
            state: 현재 Agent 상태

        Returns:
            업데이트된 상태
        """
        logger.info("Conversation Agent executing...")

        # 현재는 단순히 상태 유지
        # 필요시 대화 요약이나 맥락 정리 수행 가능

        return state

    def generate_response(self, state: AgentState) -> AgentState:
        """최종 응답 생성

        Args:
            state: 현재 Agent 상태

        Returns:
            최종 응답이 포함된 업데이트된 상태
        """
        user_input = state["user_input"]
        retrieved_docs = state.get("retrieved_docs", [])
        tool_results = state.get("tool_results", [])

        logger.info(f"Generating response for: {user_input[:50]}...")

        try:
            # 컨텍스트 포맷팅
            context = self._format_context(retrieved_docs)
            tool_results_str = self._format_tool_results(tool_results)

            # 응답 생성
            chain = self.response_prompt | self.llm
            response = chain.invoke({
                "system_prompt": self.system_prompt,
                "user_input": user_input,
                "context": context,
                "tool_results": tool_results_str,
            })

            final_response = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"Generated response: {final_response[:100]}...")

            return {
                **state,
                "final_response": final_response,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=user_input),
                    AIMessage(content=final_response),
                ],
            }

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            error_response = f"죄송합니다, 응답 생성 중 오류가 발생했습니다: {str(e)}"

            return {
                **state,
                "final_response": error_response,
                "error": str(e),
            }

    def _format_context(self, docs: List[dict]) -> str:
        """검색된 문서 포맷팅"""
        if not docs:
            return "검색된 문서가 없습니다."

        parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            score = doc.get("score", 0)

            parts.append(f"[문서 {i}] (출처: {source}, 관련도: {score:.2f})\n{content}")

        return "\n\n".join(parts)

    def _format_tool_results(self, results: List[dict]) -> str:
        """Tool 실행 결과 포맷팅"""
        if not results:
            return "Tool 실행 결과가 없습니다."

        parts = []
        for result in results:
            tool_name = result.get("tool_name", "unknown")
            success = result.get("success", False)
            data = result.get("result", {})

            status = "성공" if success else "실패"
            parts.append(f"[{tool_name}] ({status})\n{self._format_dict(data)}")

        return "\n\n".join(parts)

    def _format_dict(self, d: dict, indent: int = 0) -> str:
        """딕셔너리를 읽기 쉬운 문자열로 포맷팅"""
        if not d:
            return "결과 없음"

        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- {key}:")
                lines.append(self._format_dict(value, indent + 1))
            else:
                lines.append(f"{prefix}- {key}: {value}")

        return "\n".join(lines)


class MockConversationAgent(ConversationAgent):
    """테스트용 Mock Conversation Agent"""

    def generate_response(self, state: AgentState) -> AgentState:
        """Mock 응답 생성"""
        user_input = state["user_input"]
        retrieved_docs = state.get("retrieved_docs", [])
        tool_results = state.get("tool_results", [])

        # 간단한 응답 생성
        response_parts = [f"'{user_input}'에 대한 답변입니다."]

        if retrieved_docs:
            response_parts.append(f"\n검색된 {len(retrieved_docs)}개의 문서를 참고했습니다.")
            if retrieved_docs:
                response_parts.append(f"주요 내용: {retrieved_docs[0].get('content', '')[:100]}...")

        if tool_results:
            for result in tool_results:
                if result.get("success"):
                    tool_name = result.get("tool_name", "")
                    data = result.get("result", {})
                    response_parts.append(f"\n{tool_name} 결과: {data}")

        final_response = "\n".join(response_parts)

        return {
            **state,
            "final_response": final_response,
        }
