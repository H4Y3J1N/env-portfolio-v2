"""
Fixed-size Chunker

고정 크기 청킹 전략 구현
"""

from typing import List, Dict, Any, Optional
from .base_chunker import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """
    고정 크기 청킹

    텍스트를 고정된 단어/토큰 수로 분할합니다.
    Overlap을 통해 청크 간 문맥을 유지합니다.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.unit = config.get("unit", "words")  # words, characters

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        텍스트를 고정 크기로 분할

        Args:
            text: 분할할 텍스트
            metadata: 추가 메타데이터

        Returns:
            Chunk 리스트
        """
        metadata = metadata or {}
        chunks = []

        if self.unit == "characters":
            chunks = self._chunk_by_characters(text, metadata)
        else:
            chunks = self._chunk_by_words(text, metadata)

        return chunks

    def _chunk_by_words(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """단어 단위 청킹"""
        words = text.split()
        chunks = []

        if not words:
            return chunks

        start_idx = 0
        chunk_num = 0

        while start_idx < len(words):
            # 청크 크기만큼 단어 추출
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            # 원본 텍스트에서의 위치 계산
            start_char = self._get_char_position(words, start_idx)
            end_char = start_char + len(chunk_text)

            # Chunk 객체 생성
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunking_strategy": "fixed",
                    "chunk_index": chunk_num,
                    "word_start": start_idx,
                    "word_end": end_idx,
                },
                start_char=start_char,
                end_char=end_char,
            )

            chunks.append(chunk)
            chunk_num += 1

            # 다음 청크 시작 위치 (overlap 고려)
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size  # overlap이 너무 크면 무시
            start_idx += step

        return chunks

    def _chunk_by_characters(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """문자 단위 청킹"""
        chunks = []

        if not text:
            return chunks

        start_idx = 0
        chunk_num = 0

        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]

            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunking_strategy": "fixed",
                    "chunk_index": chunk_num,
                    "char_start": start_idx,
                    "char_end": end_idx,
                },
                start_char=start_idx,
                end_char=end_idx,
            )

            chunks.append(chunk)
            chunk_num += 1

            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size
            start_idx += step

        return chunks

    def _get_char_position(self, words: List[str], word_idx: int) -> int:
        """단어 인덱스로부터 문자 위치 계산"""
        if word_idx == 0:
            return 0

        # 이전 단어들의 총 길이 + 공백 수
        position = sum(len(w) for w in words[:word_idx]) + word_idx
        return position
