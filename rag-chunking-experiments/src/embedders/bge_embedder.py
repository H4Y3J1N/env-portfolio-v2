"""
BGE Embedder

BAAI/bge-m3 임베딩 모델 래퍼
"""

from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class BGEEmbedder:
    """
    BGE-M3 임베딩 모델

    다국어 지원, Dense + Sparse 임베딩을 제공하는 고성능 임베딩 모델입니다.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: 모델 이름 (HuggingFace Hub)
            device: 디바이스 (auto, cpu, cuda)
            normalize: 임베딩 정규화 여부
            batch_size: 배치 크기
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers가 필요합니다. "
                "pip install sentence-transformers"
            )

        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        # 디바이스 설정
        if device == "auto" or device is None:
            device = None  # SentenceTransformer가 자동 감지
        self.device = device

        # 모델 로드
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed(self, text: str) -> np.ndarray:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (1D numpy array)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding)

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        배치 텍스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트
            show_progress: 진행 바 표시 여부

        Returns:
            임베딩 행렬 (2D numpy array, shape: [N, embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        쿼리 임베딩 (검색용)

        BGE 모델의 경우 쿼리에 instruction prefix를 추가하면 성능이 향상됩니다.

        Args:
            query: 검색 쿼리

        Returns:
            쿼리 임베딩 벡터
        """
        # BGE 모델용 instruction
        instruction = "Represent this sentence for searching relevant passages: "
        query_with_instruction = instruction + query

        return self.embed(query_with_instruction)

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        문서 임베딩 (인덱싱용)

        Args:
            documents: 문서 텍스트 리스트

        Returns:
            문서 임베딩 행렬
        """
        # 문서는 instruction 없이 임베딩
        return self.embed_batch(documents)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        두 벡터 간 코사인 유사도 계산

        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터

        Returns:
            코사인 유사도 (-1 ~ 1)
        """
        if self.normalize:
            # 이미 정규화되어 있으면 내적 = 코사인 유사도
            return float(np.dot(vec1, vec2))
        else:
            # 정규화 후 내적
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def batch_similarity(
        self,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray
    ) -> np.ndarray:
        """
        쿼리와 여러 문서 간 유사도 계산

        Args:
            query_vec: 쿼리 벡터 (1D)
            doc_vecs: 문서 벡터 행렬 (2D)

        Returns:
            유사도 배열
        """
        if self.normalize:
            return np.dot(doc_vecs, query_vec)
        else:
            # 정규화 후 계산
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_vecs, axis=1)

            # 0 방지
            query_norm = max(query_norm, 1e-10)
            doc_norms = np.maximum(doc_norms, 1e-10)

            return np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

    def __repr__(self) -> str:
        return f"BGEEmbedder(model={self.model_name}, dim={self.embedding_dim})"
