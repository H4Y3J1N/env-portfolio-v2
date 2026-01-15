"""
Semantic Chunker

의미 기반 청킹 전략 구현
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .base_chunker import BaseChunker, Chunk

# Optional imports - 설치되지 않은 경우 graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


class SemanticChunker(BaseChunker):
    """
    의미 기반 청킹

    문장 임베딩 간 유사도를 측정하여 의미가 변하는 지점에서 분할합니다.
    문맥의 일관성을 최대화하지만 처리 속도가 느립니다.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers가 필요합니다. "
                "pip install sentence-transformers"
            )

        if not HAS_NLTK:
            raise ImportError(
                "nltk가 필요합니다. "
                "pip install nltk"
            )

        # NLTK 데이터 다운로드
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # 임베딩 모델 로드
        model_name = config.get("embedding_model", "BAAI/bge-m3")
        self.embedder = SentenceTransformer(model_name)

        # 설정
        self.similarity_threshold = config.get("similarity_threshold", 0.70)
        self.min_chunk_size = config.get("min_chunk_size", 200)
        self.max_chunk_size = config.get("max_chunk_size", 800)
        self.buffer_size = config.get("buffer_size", 1)
        self.use_percentile = config.get("percentile_threshold", None)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        의미 기반 분할

        Args:
            text: 분할할 텍스트
            metadata: 추가 메타데이터

        Returns:
            Chunk 리스트
        """
        metadata = metadata or {}

        # 1. 문장 분리
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [Chunk(
                text=text,
                metadata={**metadata, "chunking_strategy": "semantic"},
            )]

        # 2. 문장 임베딩
        embeddings = self.embedder.encode(sentences, show_progress_bar=False)

        # 3. 인접 문장 간 유사도 계산
        similarities = self._compute_similarities(embeddings)

        # 4. 분할 지점 결정
        split_indices = self._find_split_points(similarities)

        # 5. 청크 생성
        chunks = self._create_chunks(sentences, split_indices, text, metadata)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        # 문단 구분 보존
        paragraphs = text.split("\n\n")
        sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 문장 분리
            para_sentences = sent_tokenize(para)
            sentences.extend(para_sentences)

        return sentences

    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """인접 문장 간 코사인 유사도 계산"""
        similarities = []

        for i in range(len(embeddings) - 1):
            # 버퍼 적용 (전후 문장 평균)
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(len(embeddings), i + self.buffer_size + 1)

            # 현재 그룹 평균
            current_group = embeddings[start_idx:i + 1].mean(axis=0)

            # 다음 그룹
            next_start = i + 1
            next_end = min(len(embeddings), i + 2 + self.buffer_size)
            next_group = embeddings[next_start:next_end].mean(axis=0)

            # 코사인 유사도
            sim = self._cosine_similarity(current_group, next_group)
            similarities.append(sim)

        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _find_split_points(self, similarities: List[float]) -> List[int]:
        """분할 지점 찾기"""
        if not similarities:
            return []

        # 임계값 결정
        if self.use_percentile:
            # 백분위수 기반 임계값
            threshold = np.percentile(similarities, self.use_percentile)
        else:
            threshold = self.similarity_threshold

        # 유사도가 임계값 미만인 지점을 분할점으로
        split_indices = [0]  # 시작점

        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_indices.append(i + 1)

        return split_indices

    def _create_chunks(
        self,
        sentences: List[str],
        split_indices: List[int],
        original_text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """분할 지점을 기반으로 청크 생성"""
        # 끝점 추가
        split_indices = sorted(set(split_indices + [len(sentences)]))

        chunks = []
        chunk_idx = 0

        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]

            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            word_count = len(chunk_text.split())

            # 크기 제약 처리
            if word_count < self.min_chunk_size and chunks:
                # 너무 작으면 이전 청크와 병합
                chunks[-1].text += " " + chunk_text
                continue
            elif word_count > self.max_chunk_size:
                # 너무 크면 강제 분할
                sub_chunks = self._force_split_large(chunk_text, metadata, chunk_idx)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                continue

            # 원본 텍스트에서의 위치 찾기
            start_char = original_text.find(chunk_sentences[0]) if chunk_sentences else 0
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunking_strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "sentence_start": start,
                    "sentence_end": end,
                },
                start_char=max(0, start_char),
                end_char=end_char,
            )

            chunks.append(chunk)
            chunk_idx += 1

        return chunks

    def _force_split_large(
        self,
        text: str,
        metadata: Dict[str, Any],
        start_idx: int
    ) -> List[Chunk]:
        """큰 청크를 강제 분할"""
        words = text.split()
        chunks = []

        chunk_size = self.chunk_size
        overlap = self.chunk_overlap

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunking_strategy": "semantic",
                    "chunk_index": start_idx + len(chunks),
                    "force_split": True,
                },
            )
            chunks.append(chunk)

        return chunks
