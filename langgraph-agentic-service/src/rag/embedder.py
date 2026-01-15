"""
Hybrid Embedder

BGE-M3 기반 Dense + Sparse 임베딩
"""

import logging
from typing import List, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """BGE-M3 기반 Dense + Sparse 임베딩

    Dense: 의미적 유사성 기반 검색
    Sparse: 키워드 매칭 기반 검색 (BM25-like)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: 사용할 모델 이름
            use_fp16: FP16 사용 여부
            device: 디바이스 (cuda/cpu)
            batch_size: 배치 크기
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.batch_size = batch_size
        self.model = None

        self._load_model()

    def _load_model(self):
        """모델 로딩"""
        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info("BGE-M3 model loaded successfully")

        except ImportError:
            logger.warning("FlagEmbedding not installed. Using mock embedder.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            self.model = None

    def encode(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict[str, any]:
        """텍스트 임베딩 (Dense + Sparse)

        Args:
            texts: 임베딩할 텍스트 또는 텍스트 리스트
            return_dense: Dense 임베딩 반환 여부
            return_sparse: Sparse 임베딩 반환 여부

        Returns:
            {"dense": np.ndarray, "sparse": list[dict]} 형태의 딕셔너리
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model is None:
            # Mock 결과 반환
            return self._mock_encode(texts, return_dense, return_sparse)

        try:
            outputs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )

            result = {}

            if return_dense:
                result["dense"] = outputs["dense_vecs"]

            if return_sparse:
                result["sparse"] = outputs["lexical_weights"]

            return result

        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return self._mock_encode(texts, return_dense, return_sparse)

    def _mock_encode(
        self,
        texts: List[str],
        return_dense: bool,
        return_sparse: bool,
    ) -> Dict[str, any]:
        """Mock 임베딩 (모델 없을 때)"""
        result = {}
        n = len(texts)

        if return_dense:
            # 랜덤 dense 벡터 (1024차원)
            result["dense"] = np.random.randn(n, 1024).astype(np.float32)

        if return_sparse:
            # 간단한 sparse 표현
            result["sparse"] = [
                {hash(word) % 30000: 1.0 for word in text.split()[:10]}
                for text in texts
            ]

        return result

    def encode_query(self, query: str) -> Dict[str, any]:
        """쿼리 임베딩

        Args:
            query: 검색 쿼리

        Returns:
            임베딩 결과
        """
        return self.encode([query])

    def encode_documents(self, documents: List[str]) -> Dict[str, any]:
        """문서 임베딩

        Args:
            documents: 문서 리스트

        Returns:
            임베딩 결과
        """
        return self.encode(documents)

    @property
    def embedding_dim(self) -> int:
        """임베딩 차원"""
        return 1024  # BGE-M3 기본 차원


class MockEmbedder(HybridEmbedder):
    """테스트용 Mock Embedder"""

    def __init__(self, **kwargs):
        self.model = None
        self.batch_size = 32
        logger.info("Using MockEmbedder")

    def _load_model(self):
        pass

    def encode(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict[str, any]:
        """Mock 임베딩"""
        if isinstance(texts, str):
            texts = [texts]

        return self._mock_encode(texts, return_dense, return_sparse)

    def _mock_encode(
        self,
        texts: List[str],
        return_dense: bool,
        return_sparse: bool,
    ) -> Dict[str, any]:
        """일관된 Mock 임베딩"""
        result = {}
        n = len(texts)

        if return_dense:
            # 텍스트 해시 기반 일관된 벡터
            dense_vecs = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                vec = np.random.randn(1024).astype(np.float32)
                vec = vec / np.linalg.norm(vec)  # 정규화
                dense_vecs.append(vec)
            result["dense"] = np.array(dense_vecs)

        if return_sparse:
            result["sparse"] = [
                {hash(word) % 30000: len(word) / 10.0 for word in text.split()[:20]}
                for text in texts
            ]

        return result
