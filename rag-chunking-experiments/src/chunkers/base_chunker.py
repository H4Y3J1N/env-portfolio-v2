"""
Base Chunker

청킹 전략의 추상 기본 클래스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import time


@dataclass
class Chunk:
    """청크 데이터 클래스"""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    start_char: int = 0
    end_char: int = 0

    def __post_init__(self):
        """청크 생성 후 처리"""
        if not self.metadata:
            self.metadata = {}

    @property
    def word_count(self) -> int:
        """단어 수"""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """문자 수"""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """청킹 기본 추상 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 청킹 설정 딕셔너리
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.strategy_name = config.get("strategy", "unknown")

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            metadata: 청크에 추가할 메타데이터

        Returns:
            Chunk 리스트
        """
        pass

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Chunk]:
        """
        여러 문서를 한 번에 청킹

        Args:
            documents: 문서 리스트 [{"id": str, "text": str, "metadata": dict}, ...]
            verbose: 진행 상황 출력 여부

        Returns:
            모든 청크 리스트
        """
        all_chunks = []
        total_docs = len(documents)

        start_time = time.time()

        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{idx}")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            if not text.strip():
                continue

            # 문서 청킹
            chunks = self.chunk(text, metadata)

            # Chunk ID 및 Doc ID 할당
            for i, chunk in enumerate(chunks):
                chunk.chunk_id = f"{doc_id}_chunk_{i:04d}"
                chunk.doc_id = doc_id

            all_chunks.extend(chunks)

            if verbose:
                print(f"  [{idx + 1}/{total_docs}] {doc_id}: {len(chunks)} chunks")

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nTotal: {len(all_chunks)} chunks in {elapsed_time:.2f}s")

        return all_chunks

    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        청크 통계 계산

        Args:
            chunks: 청크 리스트

        Returns:
            통계 딕셔너리
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_size": 0,
                "std_size": 0,
                "min_size": 0,
                "max_size": 0,
                "size_uniformity": 0,
            }

        sizes = [chunk.word_count for chunk in chunks]

        avg_size = np.mean(sizes)
        std_size = np.std(sizes)

        # 크기 균일성 (1 - 변동계수)
        uniformity = 1 - (std_size / avg_size) if avg_size > 0 else 0

        return {
            "total_chunks": len(chunks),
            "avg_size": float(avg_size),
            "std_size": float(std_size),
            "min_size": int(np.min(sizes)),
            "max_size": int(np.max(sizes)),
            "median_size": float(np.median(sizes)),
            "size_uniformity": float(uniformity),
            "strategy": self.strategy_name,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
