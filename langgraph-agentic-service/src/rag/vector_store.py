"""
Vector Store

ChromaDB 기반 벡터 저장소
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB 기반 벡터 저장소

    Dense + Sparse 검색 지원
    """

    def __init__(
        self,
        collection_name: str = "agriculture_docs",
        persist_directory: str = "./data/chroma",
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 저장 디렉토리
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._init_client()

    def _init_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            import chromadb
            from chromadb.config import Settings

            # 저장 디렉토리 생성
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"ChromaDB initialized: {self.collection_name}")
            logger.info(f"Collection has {self.collection.count()} documents")

        except ImportError:
            logger.warning("chromadb not installed. Using in-memory store.")
            self._init_memory_store()
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._init_memory_store()

    def _init_memory_store(self):
        """메모리 기반 저장소 초기화 (fallback)"""
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        logger.info("Using in-memory vector store")

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """문서 추가

        Args:
            documents: 문서 텍스트 리스트
            embeddings: 임베딩 벡터 (N x D)
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        n = len(documents)

        if ids is None:
            ids = [f"doc_{i}" for i in range(n)]

        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        if self.collection is not None:
            # ChromaDB 사용
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Added {n} documents to ChromaDB")
        else:
            # 메모리 저장소 사용
            self.documents.extend(documents)
            self.embeddings.extend(embeddings.tolist())
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
            logger.info(f"Added {n} documents to memory store")

    def search_dense(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """Dense 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 문서 수
            filter: 필터 조건

        Returns:
            검색 결과 리스트
        """
        if self.collection is not None:
            # ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter,
                include=["documents", "metadatas", "distances"],
            )

            # 결과 포맷팅
            docs = []
            for i in range(len(results["ids"][0])):
                docs.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # distance → similarity
                })

            return docs

        else:
            # 메모리 저장소 검색
            return self._search_memory(query_embedding, top_k)

    def _search_memory(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict]:
        """메모리 저장소 검색"""
        if not self.embeddings:
            return []

        # Cosine similarity 계산
        embeddings = np.array(self.embeddings)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)

        # Top-k 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]

        docs = []
        for idx in top_indices:
            docs.append({
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(similarities[idx]),
            })

        return docs

    def search_sparse(
        self,
        sparse_vector: Dict[int, float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """Sparse 검색 (BM25-like)

        Note: ChromaDB는 native sparse search를 지원하지 않으므로
              여기서는 간단한 키워드 매칭으로 대체

        Args:
            sparse_vector: sparse 가중치 딕셔너리
            top_k: 반환할 문서 수
            filter: 필터 조건

        Returns:
            검색 결과 리스트
        """
        # ChromaDB에서 모든 문서 가져오기 (작은 컬렉션 가정)
        if self.collection is not None:
            all_docs = self.collection.get(include=["documents", "metadatas"])

            if not all_docs["ids"]:
                return []

            # 간단한 키워드 스코어링
            scores = []
            query_tokens = set(sparse_vector.keys())

            for doc in all_docs["documents"]:
                # 문서 토큰화 (간단한 방식)
                doc_tokens = set(hash(word) % 30000 for word in doc.split())
                # 공통 토큰 수로 스코어 계산
                common = len(query_tokens & doc_tokens)
                scores.append(common)

            # Top-k 선택
            indices = np.argsort(scores)[::-1][:top_k]

            docs = []
            for idx in indices:
                docs.append({
                    "id": all_docs["ids"][idx],
                    "content": all_docs["documents"][idx],
                    "metadata": all_docs["metadatas"][idx],
                    "score": float(scores[idx]),
                })

            return docs

        else:
            # 메모리 저장소 sparse 검색
            return self._sparse_search_memory(sparse_vector, top_k)

    def _sparse_search_memory(
        self,
        sparse_vector: Dict[int, float],
        top_k: int,
    ) -> List[Dict]:
        """메모리 저장소 sparse 검색"""
        if not self.documents:
            return []

        query_tokens = set(sparse_vector.keys())
        scores = []

        for doc in self.documents:
            doc_tokens = set(hash(word) % 30000 for word in doc.split())
            common = len(query_tokens & doc_tokens)
            scores.append(common)

        indices = np.argsort(scores)[::-1][:top_k]

        docs = []
        for idx in indices:
            docs.append({
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[idx]),
            })

        return docs

    def count(self) -> int:
        """문서 수 반환"""
        if self.collection is not None:
            return self.collection.count()
        return len(self.documents)

    def delete_collection(self):
        """컬렉션 삭제"""
        if self.client is not None:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
