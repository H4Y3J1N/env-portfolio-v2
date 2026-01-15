"""
RAG Agent

Hybrid RAG 검색을 수행하는 Agent
"""

import logging
from typing import Optional
from langchain_core.messages import AIMessage

from .state import AgentState

logger = logging.getLogger(__name__)


class RAGAgent:
    """RAG 검색 Agent

    Hybrid RAG 파이프라인을 사용하여 관련 문서 검색
    """

    def __init__(
        self,
        rag_pipeline=None,
        top_k: int = 5,
    ):
        """
        Args:
            rag_pipeline: HybridRAGPipeline 인스턴스
            top_k: 검색할 문서 수
        """
        self.rag_pipeline = rag_pipeline
        self.top_k = top_k

    def execute(self, state: AgentState) -> AgentState:
        """RAG 검색 실행

        Args:
            state: 현재 Agent 상태

        Returns:
            검색 결과가 포함된 업데이트된 상태
        """
        user_input = state["user_input"]
        logger.info(f"RAG Agent executing for query: {user_input[:50]}...")

        try:
            if self.rag_pipeline is None:
                # RAG 파이프라인이 없으면 더미 결과 반환
                logger.warning("RAG pipeline not configured, returning empty results")
                return {
                    **state,
                    "retrieved_docs": [],
                }

            # Hybrid RAG 검색
            documents = self.rag_pipeline.search(
                query=user_input,
                top_k=self.top_k,
            )

            # 문서를 딕셔너리 형태로 변환
            retrieved_docs = [
                {
                    "content": doc.page_content if hasattr(doc, 'page_content') else doc.get("content", ""),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else doc.get("metadata", {}),
                    "score": doc.metadata.get("rerank_score", 0) if hasattr(doc, 'metadata') else doc.get("score", 0),
                }
                for doc in documents
            ]

            logger.info(f"RAG Agent found {len(retrieved_docs)} documents")

            # 상태 업데이트
            return {
                **state,
                "retrieved_docs": retrieved_docs,
            }

        except Exception as e:
            logger.error(f"RAG Agent error: {e}")
            return {
                **state,
                "retrieved_docs": [],
                "error": f"RAG 검색 오류: {str(e)}",
            }

    def format_context(self, docs: list[dict]) -> str:
        """검색된 문서를 컨텍스트 문자열로 포맷팅

        Args:
            docs: 검색된 문서 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        if not docs:
            return "검색된 문서가 없습니다."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            score = doc.get("score", 0)

            context_parts.append(
                f"[문서 {i}] (출처: {source}, 관련도: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)


class MockRAGAgent(RAGAgent):
    """테스트용 Mock RAG Agent"""

    def __init__(self):
        super().__init__(rag_pipeline=None)
        self.mock_docs = [
            {
                "content": "토마토 수확량은 재식밀도, 기온, 일조량에 따라 달라집니다. 일반적으로 10a당 5-8톤의 수확량을 기대할 수 있습니다.",
                "metadata": {"source": "agriculture_manual.pdf", "page": 15},
                "score": 0.92,
            },
            {
                "content": "영농형 태양광 시스템에서 패널 각도는 작물 생육에 큰 영향을 미칩니다. 계절별로 30-45도 범위에서 조정하는 것이 권장됩니다.",
                "metadata": {"source": "solar_farming_guide.pdf", "page": 8},
                "score": 0.85,
            },
            {
                "content": "기온 예측 모델은 과거 데이터와 기상청 API를 활용합니다. 주간 예보의 정확도는 약 85% 수준입니다.",
                "metadata": {"source": "weather_prediction.pdf", "page": 3},
                "score": 0.78,
            },
        ]

    def execute(self, state: AgentState) -> AgentState:
        """Mock 검색 실행"""
        user_input = state["user_input"].lower()

        # 키워드 기반 간단한 매칭
        relevant_docs = []
        for doc in self.mock_docs:
            content_lower = doc["content"].lower()
            if any(keyword in content_lower for keyword in user_input.split()):
                relevant_docs.append(doc)

        # 매칭된 문서가 없으면 첫 번째 문서 반환
        if not relevant_docs:
            relevant_docs = self.mock_docs[:1]

        logger.info(f"Mock RAG Agent found {len(relevant_docs)} documents")

        return {
            **state,
            "retrieved_docs": relevant_docs,
        }
