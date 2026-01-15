"""
Hybrid Retriever

BM25 + Dense 검색 하이브리드 구현
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .vector_retriever import VectorRetriever, SearchResult

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from chunkers.base_chunker import Chunk
from embedders.bge_embedder import BGEEmbedder


class HybridRetriever:
    """
    하이브리드 검색기

    BM25 (키워드 기반)와 Dense (의미 기반) 검색을 결합합니다.
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """
        Args:
            embedder: 임베딩 모델
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse (BM25) 검색 가중치
        """
        if not HAS_BM25:
            raise ImportError(
                "rank_bm25가 필요합니다. "
                "pip install rank-bm25"
            )

        self.embedder = embedder or BGEEmbedder()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Dense retriever
        self.vector_retriever = VectorRetriever(self.embedder)

        # BM25
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[np.ndarray] = None
    ):
        """
        청크 추가 및 인덱싱

        Args:
            chunks: 청크 리스트
            embeddings: 사전 계산된 임베딩
        """
        self.chunks = chunks

        # Dense 인덱싱
        self.vector_retriever.add_chunks(chunks, embeddings)

        # BM25 인덱싱
        print("Building BM25 index...")
        self.tokenized_corpus = [
            self._tokenize(chunk.text) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"Indexed {len(chunks)} chunks (hybrid)")

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토크나이저"""
        # 소문자 변환 및 단어 분리
        text = text.lower()
        # 특수문자 제거
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def search(
        self,
        query: str,
        k: int = 5,
        use_rrf: bool = True
    ) -> List[SearchResult]:
        """
        하이브리드 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            use_rrf: Reciprocal Rank Fusion 사용 여부

        Returns:
            SearchResult 리스트
        """
        # Dense 검색
        dense_results = self.vector_retriever.search(query, k * 2)

        # BM25 검색
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # BM25 상위 k*2 추출
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]

        if use_rrf:
            # Reciprocal Rank Fusion
            return self._rrf_fusion(dense_results, top_bm25_indices, bm25_scores, k)
        else:
            # 가중 합산
            return self._weighted_fusion(dense_results, bm25_scores, k)

    def _rrf_fusion(
        self,
        dense_results: List[SearchResult],
        bm25_indices: np.ndarray,
        bm25_scores: np.ndarray,
        k: int,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion"""
        scores = defaultdict(float)

        # Dense 점수
        for rank, result in enumerate(dense_results, 1):
            scores[result.chunk_id] += self.dense_weight / (rrf_k + rank)

        # BM25 점수
        for rank, idx in enumerate(bm25_indices, 1):
            chunk_id = self.chunks[idx].chunk_id
            scores[chunk_id] += self.sparse_weight / (rrf_k + rank)

        # 정렬
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

        # 결과 생성
        results = []
        for rank, chunk_id in enumerate(sorted_ids, 1):
            chunk = self.vector_retriever.get_chunk_by_id(chunk_id)
            if chunk:
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=scores[chunk_id],
                    rank=rank,
                    metadata=chunk.metadata,
                ))

        return results

    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        bm25_scores: np.ndarray,
        k: int
    ) -> List[SearchResult]:
        """가중 합산 퓨전"""
        scores = {}

        # Dense 점수 정규화
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            min_dense = min(r.score for r in dense_results)
            dense_range = max_dense - min_dense if max_dense != min_dense else 1

            for result in dense_results:
                norm_score = (result.score - min_dense) / dense_range
                scores[result.chunk_id] = self.dense_weight * norm_score

        # BM25 점수 정규화
        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 1
        min_bm25 = np.min(bm25_scores) if len(bm25_scores) > 0 else 0
        bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1

        for idx, score in enumerate(bm25_scores):
            chunk_id = self.chunks[idx].chunk_id
            norm_score = (score - min_bm25) / bm25_range
            scores[chunk_id] = scores.get(chunk_id, 0) + self.sparse_weight * norm_score

        # 정렬
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

        # 결과 생성
        results = []
        for rank, chunk_id in enumerate(sorted_ids, 1):
            chunk = self.vector_retriever.get_chunk_by_id(chunk_id)
            if chunk:
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=scores[chunk_id],
                    rank=rank,
                    metadata=chunk.metadata,
                ))

        return results

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"HybridRetriever(chunks={len(self.chunks)}, dense_w={self.dense_weight}, sparse_w={self.sparse_weight})"
