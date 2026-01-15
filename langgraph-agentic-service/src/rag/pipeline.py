"""
Hybrid RAG Pipeline

전체 RAG 파이프라인 통합
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from langchain_core.documents import Document

from .embedder import HybridEmbedder, MockEmbedder
from .vector_store import ChromaVectorStore
from .retriever import HybridRetriever
from .reranker import CrossEncoderReranker, MockReranker

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    """Hybrid RAG 파이프라인

    BGE-M3 (Dense + Sparse) + Cross-encoder Reranker
    """

    def __init__(
        self,
        embedder: Optional[HybridEmbedder] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
        use_mock: bool = False,
    ):
        """
        Args:
            embedder: 임베딩 모델
            vector_store: 벡터 저장소
            reranker: 리랭커
            retrieve_top_k: 검색할 문서 수
            rerank_top_k: 리랭킹 후 반환할 문서 수
            use_mock: Mock 모드 사용 여부
        """
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

        if use_mock:
            logger.info("Initializing RAG pipeline in mock mode")
            self.embedder = MockEmbedder()
            self.vector_store = ChromaVectorStore(
                collection_name="mock_collection",
                persist_directory="./data/mock_chroma",
            )
            self.reranker = MockReranker()
        else:
            self.embedder = embedder or HybridEmbedder()
            self.vector_store = vector_store or ChromaVectorStore()
            self.reranker = reranker or CrossEncoderReranker()

        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """RAG 검색 (Retrieve → Rerank)

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수 (기본: rerank_top_k)
            filter_metadata: 메타데이터 필터

        Returns:
            검색된 Document 리스트
        """
        top_k = top_k or self.rerank_top_k

        logger.info(f"RAG search for: {query[:50]}...")

        # 1. Hybrid Retrieval (Dense + Sparse)
        retrieved = self.retriever.retrieve(
            query=query,
            top_k=self.retrieve_top_k,
            filter_metadata=filter_metadata,
        )

        if not retrieved:
            logger.warning("No documents retrieved")
            return []

        logger.info(f"Retrieved {len(retrieved)} documents")

        # 2. Cross-encoder Reranking
        reranked = self.reranker.rerank(
            query=query,
            documents=retrieved,
            top_k=top_k,
        )

        logger.info(f"Reranked to {len(reranked)} documents")

        # 3. Document 형식으로 변환
        documents = [
            Document(
                page_content=doc.get("content", ""),
                metadata={
                    **doc.get("metadata", {}),
                    "id": doc.get("id", ""),
                    "rrf_score": doc.get("rrf_score", 0),
                    "rerank_score": doc.get("rerank_score", 0),
                },
            )
            for doc in reranked
        ]

        return documents

    def add_documents(
        self,
        documents: Union[List[str], List[Document]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """문서 인덱싱

        Args:
            documents: 문서 텍스트 또는 Document 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        # Document 객체 처리
        if documents and isinstance(documents[0], Document):
            texts = [doc.page_content for doc in documents]
            if metadatas is None:
                metadatas = [doc.metadata for doc in documents]
        else:
            texts = documents

        self.retriever.add_documents(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def add_documents_from_texts(
        self,
        texts: List[str],
        source: str = "unknown",
    ):
        """텍스트 리스트로 문서 추가

        Args:
            texts: 텍스트 리스트
            source: 출처 정보
        """
        metadatas = [{"source": source, "chunk_id": i} for i in range(len(texts))]
        ids = [f"{source}_{i}" for i in range(len(texts))]

        self.add_documents(texts, metadatas, ids)

    def count_documents(self) -> int:
        """인덱싱된 문서 수"""
        return self.vector_store.count()

    @classmethod
    def create_mock_pipeline(cls) -> "HybridRAGPipeline":
        """Mock 파이프라인 생성"""
        pipeline = cls(use_mock=True)

        # 샘플 문서 추가
        sample_docs = [
            "토마토 수확량은 재식밀도, 기온, 일조량에 따라 달라집니다. 일반적으로 10a당 5-8톤의 수확량을 기대할 수 있습니다. 적정 재식밀도는 평당 4-5주이며, 여름철 고온기에는 차광막을 설치하여 온도를 조절해야 합니다.",
            "영농형 태양광 시스템에서 패널 각도는 작물 생육에 큰 영향을 미칩니다. 계절별로 30-45도 범위에서 조정하는 것이 권장됩니다. 봄과 가을에는 35도, 여름에는 45도, 겨울에는 30도가 적정합니다.",
            "기온 예측 모델은 과거 데이터와 기상청 API를 활용합니다. 주간 예보의 정확도는 약 85% 수준입니다. 농업용 기온 예측은 서리 예방과 관수 시기 결정에 중요하게 활용됩니다.",
            "고추 재배 시 적정 온도는 주간 25-28도, 야간 18-22도입니다. 온도가 35도 이상이면 낙화가 발생하고, 15도 이하에서는 생육이 저하됩니다.",
            "배추는 서늘한 기후를 좋아하는 작물로, 생육 적온은 15-20도입니다. 고온기 재배 시 뿌리혹병 발생에 주의해야 합니다.",
        ]

        pipeline.add_documents_from_texts(sample_docs, source="sample_manual")
        logger.info(f"Mock pipeline created with {len(sample_docs)} documents")

        return pipeline


def create_rag_pipeline(
    config: Optional[Dict] = None,
    use_mock: bool = False,
) -> HybridRAGPipeline:
    """RAG 파이프라인 팩토리 함수

    Args:
        config: 설정 딕셔너리
        use_mock: Mock 모드 사용 여부

    Returns:
        HybridRAGPipeline 인스턴스
    """
    if use_mock:
        return HybridRAGPipeline.create_mock_pipeline()

    config = config or {}

    embedder_config = config.get("embedder", {})
    reranker_config = config.get("reranker", {})
    vector_store_config = config.get("vector_store", {})

    embedder = HybridEmbedder(
        model_name=embedder_config.get("model_name", "BAAI/bge-m3"),
        device=embedder_config.get("device", "cuda"),
        use_fp16=embedder_config.get("use_fp16", True),
    )

    vector_store = ChromaVectorStore(
        collection_name=vector_store_config.get("collection_name", "agriculture_docs"),
        persist_directory=vector_store_config.get("persist_directory", "./data/chroma"),
    )

    reranker = CrossEncoderReranker(
        model_name=reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3"),
        device=reranker_config.get("device", "cuda"),
    )

    return HybridRAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        retrieve_top_k=config.get("retrieve_top_k", 20),
        rerank_top_k=config.get("rerank_top_k", 5),
    )
