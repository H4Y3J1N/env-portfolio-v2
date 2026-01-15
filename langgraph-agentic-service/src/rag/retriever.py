"""
Hybrid Retriever

Dense + Sparse 하이브리드 검색
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from .embedder import HybridEmbedder
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + Sparse Hybrid 검색기

    RRF (Reciprocal Rank Fusion)를 사용하여 Dense와 Sparse 결과 통합
    """

    def __init__(
        self,
        embedder: HybridEmbedder,
        vector_store: ChromaVectorStore,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """
        Args:
            embedder: 임베딩 모델
            vector_store: 벡터 저장소
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse 검색 가중치
            rrf_k: RRF 파라미터
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """하이브리드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            filter_metadata: 메타데이터 필터

        Returns:
            검색 결과 리스트
        """
        logger.info(f"Hybrid retrieval for: {query[:50]}...")

        # 1. 쿼리 임베딩 (Dense + Sparse)
        query_embedding = self.embedder.encode(
            query,
            return_dense=True,
            return_sparse=True,
        )

        dense_vec = query_embedding["dense"][0]
        sparse_vec = query_embedding["sparse"][0]

        # 2. Dense 검색
        dense_results = self.vector_store.search_dense(
            dense_vec,
            top_k=top_k * 2,  # 여유있게 검색
            filter=filter_metadata,
        )
        logger.debug(f"Dense search returned {len(dense_results)} results")

        # 3. Sparse 검색
        sparse_results = self.vector_store.search_sparse(
            sparse_vec,
            top_k=top_k * 2,
            filter=filter_metadata,
        )
        logger.debug(f"Sparse search returned {len(sparse_results)} results")

        # 4. RRF로 결과 통합
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
        )

        logger.info(f"Hybrid retrieval returned {len(fused_results[:top_k])} results")

        return fused_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion) 기반 점수 통합

        RRF Score = sum(1 / (k + rank))

        Args:
            dense_results: Dense 검색 결과
            sparse_results: Sparse 검색 결과

        Returns:
            통합된 결과 리스트
        """
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict] = {}

        # Dense 결과 점수 계산
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc

        # Sparse 결과 점수 계산
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # 점수 기준 정렬
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids:
            doc = doc_data[doc_id].copy()
            doc["rrf_score"] = doc_scores[doc_id]
            results.append(doc)

        return results

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """문서 추가

        Args:
            documents: 문서 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        logger.info(f"Adding {len(documents)} documents to index")

        # 임베딩 생성
        embeddings = self.embedder.encode_documents(documents)
        dense_vecs = embeddings["dense"]

        # 벡터 저장소에 추가
        self.vector_store.add_documents(
            documents=documents,
            embeddings=dense_vecs,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Successfully added {len(documents)} documents")


class MockRetriever(HybridRetriever):
    """테스트용 Mock Retriever"""

    def __init__(self):
        self.mock_docs = [
            {
                "id": "doc_1",
                "content": "토마토 수확량은 재식밀도, 기온, 일조량에 따라 달라집니다. 일반적으로 10a당 5-8톤의 수확량을 기대할 수 있습니다.",
                "metadata": {"source": "agriculture_manual.pdf", "page": 15},
                "score": 0.92,
            },
            {
                "id": "doc_2",
                "content": "영농형 태양광 시스템에서 패널 각도는 작물 생육에 큰 영향을 미칩니다. 계절별로 30-45도 범위에서 조정하는 것이 권장됩니다.",
                "metadata": {"source": "solar_farming_guide.pdf", "page": 8},
                "score": 0.85,
            },
            {
                "id": "doc_3",
                "content": "기온 예측 모델은 과거 데이터와 기상청 API를 활용합니다. 주간 예보의 정확도는 약 85% 수준입니다.",
                "metadata": {"source": "weather_prediction.pdf", "page": 3},
                "score": 0.78,
            },
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """Mock 검색"""
        query_lower = query.lower()

        # 키워드 기반 간단한 매칭
        results = []
        for doc in self.mock_docs:
            content_lower = doc["content"].lower()
            match_score = sum(
                1 for keyword in query_lower.split()
                if keyword in content_lower
            )
            if match_score > 0:
                doc_copy = doc.copy()
                doc_copy["rrf_score"] = match_score * 0.1
                results.append(doc_copy)

        # 매칭 결과가 없으면 첫 번째 문서 반환
        if not results:
            doc_copy = self.mock_docs[0].copy()
            doc_copy["rrf_score"] = 0.5
            results.append(doc_copy)

        return sorted(results, key=lambda x: x["rrf_score"], reverse=True)[:top_k]
