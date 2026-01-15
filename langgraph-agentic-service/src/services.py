"""
Service Layer

Agent 및 RAG 서비스 통합
"""

import logging
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator

from .agents import (
    AgentState,
    SupervisorAgent,
    RAGAgent,
    ToolAgent,
    ConversationAgent,
    MockRAGAgent,
    create_agent_graph,
    create_simple_graph,
)
from .agents.state import create_initial_state
from .rag import HybridRAGPipeline
from .session import RedisSessionStore, ConversationMemory
from .monitoring import LangfuseTracker

logger = logging.getLogger(__name__)


class AgentService:
    """Multi-Agent 서비스

    LangGraph 기반 Agent 오케스트레이션
    """

    def __init__(
        self,
        use_rag: bool = True,
        use_tools: bool = True,
        max_iterations: int = 5,
        use_mock: bool = False,
    ):
        """
        Args:
            use_rag: RAG 사용 여부
            use_tools: Tool 사용 여부
            max_iterations: 최대 반복 횟수
            use_mock: Mock 모드 사용 여부
        """
        self.use_rag = use_rag
        self.use_tools = use_tools
        self.max_iterations = max_iterations
        self.use_mock = use_mock

        self.session_store = RedisSessionStore()
        self.memory = ConversationMemory(self.session_store)
        self.tracker = LangfuseTracker()

        self._graph = None
        self._initialize_graph()

    def _initialize_graph(self):
        """Agent 그래프 초기화"""
        logger.info(f"Initializing agent graph (mock={self.use_mock})")

        if self.use_mock:
            # Mock 모드
            rag_agent = MockRAGAgent()
            tool_agent = ToolAgent()
            conversation_agent = ConversationAgent()

            # 간단한 그래프 사용
            self._graph = create_simple_graph(rag_agent, conversation_agent)

        else:
            # 실제 모드
            try:
                rag_pipeline = HybridRAGPipeline.create_mock_pipeline()
                rag_agent = RAGAgent(rag_pipeline=rag_pipeline)
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}. Using mock.")
                rag_agent = MockRAGAgent()

            supervisor = SupervisorAgent(max_iterations=self.max_iterations)
            tool_agent = ToolAgent()
            conversation_agent = ConversationAgent()

            self._graph = create_agent_graph(
                supervisor=supervisor,
                rag_agent=rag_agent,
                tool_agent=tool_agent,
                conversation_agent=conversation_agent,
            )

        logger.info("Agent graph initialized")

    async def execute(
        self,
        message: str,
        session_id: str = "default",
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Agent 실행 (동기)

        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            tenant_id: 테넌트 ID

        Returns:
            실행 결과 딕셔너리
        """
        logger.info(f"Executing agent for: {message[:50]}...")

        # 트레이스 생성
        trace = self.tracker.create_trace(
            name="agent_execution",
            session_id=session_id,
        )

        try:
            # 세션 확인/생성
            await self.session_store.get_or_create(session_id, tenant_id)

            # 초기 상태 생성
            initial_state = create_initial_state(
                user_input=message,
                session_id=session_id,
                tenant_id=tenant_id,
            )

            # 그래프 실행
            result = self._graph.invoke(initial_state)

            # 결과 추출
            final_response = result.get("final_response", "")
            retrieved_docs = result.get("retrieved_docs", [])
            tool_results = result.get("tool_results", [])

            # 대화 저장
            await self.memory.add_interaction(
                session_id=session_id,
                user_message=message,
                assistant_response=final_response,
            )

            # 추적
            self.tracker.track_retrieval(
                trace, "rag_search", message, retrieved_docs
            )

            logger.info(f"Agent execution completed: {final_response[:50]}...")

            return {
                "response": final_response,
                "retrieved_docs": retrieved_docs,
                "tool_results": tool_results,
                "iterations": result.get("iteration_count", 0),
                "agents_used": self._extract_agents_used(result),
                "reasoning_trace": [result.get("reasoning", "")],
            }

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {
                "response": f"죄송합니다, 오류가 발생했습니다: {str(e)}",
                "error": str(e),
            }

        finally:
            self.tracker.flush()

    async def execute_stream(
        self,
        message: str,
        session_id: str = "default",
        tenant_id: str = "default",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Agent 실행 (스트리밍)

        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            tenant_id: 테넌트 ID

        Yields:
            이벤트 딕셔너리
        """
        logger.info(f"Streaming agent execution for: {message[:50]}...")

        try:
            # 세션 확인/생성
            await self.session_store.get_or_create(session_id, tenant_id)

            # Agent 시작 이벤트
            yield {
                "type": "agent_start",
                "agent": "supervisor",
                "reasoning": "요청 분석 중...",
            }

            await asyncio.sleep(0.1)  # 시뮬레이션 딜레이

            # 초기 상태 생성
            initial_state = create_initial_state(
                user_input=message,
                session_id=session_id,
                tenant_id=tenant_id,
            )

            # RAG 결과 이벤트
            yield {
                "type": "agent_start",
                "agent": "rag",
                "reasoning": "관련 문서 검색 중...",
            }

            await asyncio.sleep(0.2)

            # 그래프 실행
            result = self._graph.invoke(initial_state)

            # RAG 결과 스트리밍
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                yield {
                    "type": "rag_result",
                    "documents": retrieved_docs,
                }

            await asyncio.sleep(0.1)

            # Tool 결과 스트리밍
            tool_results = result.get("tool_results", [])
            for tool_result in tool_results:
                yield {
                    "type": "tool_call",
                    "tool": tool_result.get("tool_name", "unknown"),
                    "arguments": tool_result.get("arguments", {}),
                    "result": tool_result.get("result"),
                    "success": tool_result.get("success", True),
                }
                await asyncio.sleep(0.1)

            # 응답 생성 이벤트
            yield {
                "type": "agent_start",
                "agent": "conversation",
                "reasoning": "응답 생성 중...",
            }

            await asyncio.sleep(0.2)

            # 최종 응답 (토큰 단위 시뮬레이션)
            final_response = result.get("final_response", "응답을 생성할 수 없습니다.")

            # 토큰 단위로 스트리밍 (시뮬레이션)
            words = final_response.split()
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield {
                    "type": "token",
                    "token": token,
                }
                await asyncio.sleep(0.02)  # 토큰 딜레이

            # 대화 저장
            await self.memory.add_interaction(
                session_id=session_id,
                user_message=message,
                assistant_response=final_response,
            )

            # 완료 이벤트
            yield {
                "type": "response",
                "content": final_response,
                "complete": True,
            }

        except Exception as e:
            logger.error(f"Stream execution error: {e}")
            yield {
                "type": "error",
                "error": str(e),
            }

    def _extract_agents_used(self, result: Dict) -> list:
        """사용된 Agent 목록 추출"""
        agents = []
        if result.get("retrieved_docs"):
            agents.append("rag")
        if result.get("tool_results"):
            agents.append("tool")
        agents.append("conversation")
        return agents


async def initialize_services():
    """서비스 초기화 (앱 시작 시)"""
    logger.info("Initializing services...")

    # Redis 연결 테스트
    try:
        store = RedisSessionStore()
        await store.ping()
        logger.info("Redis connection: OK")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")

    # RAG 파이프라인 초기화
    try:
        pipeline = HybridRAGPipeline.create_mock_pipeline()
        logger.info(f"RAG pipeline: OK ({pipeline.count_documents()} docs)")
    except Exception as e:
        logger.warning(f"RAG pipeline initialization failed: {e}")

    logger.info("Services initialized")
