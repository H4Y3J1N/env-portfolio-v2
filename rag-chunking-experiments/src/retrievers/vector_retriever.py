"""
Vector Retriever

FAISS 기반 벡터 검색 구현
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import sys
sys.path.append(str(Path(__file__).parent.parent))

from chunkers.base_chunker import Chunk
from embedders.bge_embedder import BGEEmbedder


@dataclass
class SearchResult:
    """검색 결과"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class VectorRetriever:
    """
    벡터 기반 검색기

    FAISS를 사용한 고속 벡터 검색을 제공합니다.
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        similarity_metric: str = "cosine"
    ):
        """
        Args:
            embedder: 임베딩 모델 (없으면 자동 생성)
            similarity_metric: 유사도 메트릭 (cosine, euclidean, dot_product)
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss가 필요합니다. "
                "pip install faiss-cpu 또는 pip install faiss-gpu"
            )

        self.embedder = embedder or BGEEmbedder()
        self.similarity_metric = similarity_metric

        # 인덱스 및 청크 저장소
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[np.ndarray] = None
    ):
        """
        청크 추가 및 인덱싱

        Args:
            chunks: 청크 리스트
            embeddings: 사전 계산된 임베딩 (없으면 자동 생성)
        """
        if not chunks:
            return

        # 임베딩 생성
        if embeddings is None:
            print("Generating embeddings...")
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed_documents(texts)

        embeddings = np.array(embeddings).astype('float32')

        # 인덱스 생성
        dim = embeddings.shape[1]

        if self.similarity_metric == "cosine":
            # L2 정규화 후 Inner Product = Cosine Similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dim)
        elif self.similarity_metric == "euclidean":
            self.index = faiss.IndexFlatL2(dim)
        else:  # dot_product
            self.index = faiss.IndexFlatIP(dim)

        # 인덱스에 추가
        self.index.add(embeddings)

        # 청크 저장
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)

        for i, chunk in enumerate(chunks):
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i

        print(f"Indexed {len(chunks)} chunks (total: {len(self.chunks)})")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        쿼리 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_doc_ids: 필터링할 문서 ID 리스트

        Returns:
            SearchResult 리스트
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        # 쿼리 임베딩
        query_vec = self.embedder.embed_query(query)
        query_vec = np.array([query_vec]).astype('float32')

        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_vec)

        # 검색 (필터링 고려하여 더 많이 검색)
        search_k = k * 3 if filter_doc_ids else k
        scores, indices = self.index.search(query_vec, min(search_k, len(self.chunks)))

        # 결과 변환
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # 문서 ID 필터링
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue

            result = SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(score),
                rank=len(results) + 1,
                metadata=chunk.metadata,
            )
            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_batch(
        self,
        queries: List[str],
        k: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        배치 쿼리 검색

        Args:
            queries: 쿼리 리스트
            k: 각 쿼리당 반환할 결과 수

        Returns:
            {query: [SearchResult, ...], ...}
        """
        results = {}

        for query in queries:
            results[query] = self.search(query, k)

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """ID로 청크 조회"""
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None and idx < len(self.chunks):
            return self.chunks[idx]
        return None

    def save(self, path: str):
        """인덱스 및 청크 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 저장
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # 청크 메타데이터 저장
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {path}")

    def load(self, path: str):
        """인덱스 및 청크 로드"""
        path = Path(path)

        # FAISS 인덱스 로드
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        # 청크 메타데이터 로드
        chunks_path = path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            self.chunks = []
            self.chunk_id_to_idx = {}

            for i, data in enumerate(chunks_data):
                chunk = Chunk(
                    text=data["text"],
                    chunk_id=data["chunk_id"],
                    doc_id=data["doc_id"],
                    metadata=data.get("metadata", {}),
                    start_char=data.get("start_char", 0),
                    end_char=data.get("end_char", 0),
                )
                self.chunks.append(chunk)
                self.chunk_id_to_idx[chunk.chunk_id] = i

        print(f"Loaded {len(self.chunks)} chunks from {path}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return f"VectorRetriever(chunks={len(self.chunks)}, metric={self.similarity_metric})"
