"""
Cross-encoder Reranker

BGE-reranker 기반 문서 리랭킹
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder 기반 Reranker

    Query-Document 쌍의 관련도를 직접 계산하여 리랭킹
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: 사용할 reranker 모델
            device: 디바이스 (cuda/cpu)
            max_length: 최대 입력 길이
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None

        self._load_model()

    def _load_model(self):
        """모델 로딩"""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.info("Reranker model loaded successfully")

        except ImportError:
            logger.warning("sentence-transformers not installed. Using mock reranker.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """문서 리랭킹

        Args:
            query: 검색 쿼리
            documents: 리랭킹할 문서 리스트
            top_k: 반환할 문서 수

        Returns:
            리랭킹된 문서 리스트
        """
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")

        if self.model is None:
            # Mock 리랭킹 (기존 순서 유지)
            return self._mock_rerank(documents, top_k)

        try:
            # Query-Document 쌍 생성
            pairs = [
                [query, doc.get("content", "")]
                for doc in documents
            ]

            # Cross-encoder 점수 계산
            scores = self.model.predict(pairs)

            # 점수 추가
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # 점수 기준 정렬
            reranked = sorted(
                documents,
                key=lambda x: x["rerank_score"],
                reverse=True,
            )

            logger.info(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return self._mock_rerank(documents, top_k)

    def _mock_rerank(self, documents: List[Dict], top_k: int) -> List[Dict]:
        """Mock 리랭킹"""
        # 기존 점수 기반으로 가짜 rerank_score 생성
        for i, doc in enumerate(documents):
            existing_score = doc.get("rrf_score", doc.get("score", 0))
            doc["rerank_score"] = existing_score * 0.9 - i * 0.01

        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    def compute_score(self, query: str, document: str) -> float:
        """단일 Query-Document 쌍의 점수 계산

        Args:
            query: 검색 쿼리
            document: 문서 텍스트

        Returns:
            관련도 점수
        """
        if self.model is None:
            return 0.5  # Mock score

        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Score computation error: {e}")
            return 0.5


class MockReranker(CrossEncoderReranker):
    """테스트용 Mock Reranker"""

    def __init__(self, **kwargs):
        self.model = None
        logger.info("Using MockReranker")

    def _load_model(self):
        pass

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Mock 리랭킹 - 키워드 매칭 기반"""
        query_words = set(query.lower().split())

        for doc in documents:
            content = doc.get("content", "").lower()
            content_words = set(content.split())

            # 공통 단어 수 기반 점수
            common_words = query_words & content_words
            doc["rerank_score"] = len(common_words) / (len(query_words) + 1)

        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]
