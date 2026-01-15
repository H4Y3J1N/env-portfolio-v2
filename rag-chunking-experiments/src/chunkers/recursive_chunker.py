"""
Recursive Chunker

재귀적 청킹 전략 구현
"""

from typing import List, Dict, Any, Optional
import re
from .base_chunker import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """
    재귀적 청킹

    계층적 구분자를 사용하여 문서 구조를 보존하면서 분할합니다.
    큰 구분자(섹션)부터 시작하여 작은 구분자(문장, 단어)로 재귀적으로 분할합니다.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # 구분자 우선순위 (큰 것 -> 작은 것)
        self.separators = config.get("separators", [
            "\n\n\n",   # 섹션
            "\n\n",     # 문단
            "\n",       # 줄
            ". ",       # 문장
            "? ",       # 질문
            "! ",       # 감탄
            "; ",       # 세미콜론
            ", ",       # 쉼표
            " ",        # 단어
        ])

        self.keep_separator = config.get("keep_separator", True)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        텍스트를 재귀적으로 분할

        Args:
            text: 분할할 텍스트
            metadata: 추가 메타데이터

        Returns:
            Chunk 리스트
        """
        metadata = metadata or {}

        # 재귀 분할 실행
        text_chunks = self._split_text(text, self.separators)

        # Chunk 객체로 변환
        chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # 원본 텍스트에서의 위치 찾기
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunking_strategy": "recursive",
                    "chunk_index": i,
                },
                start_char=start_char,
                end_char=end_char,
            )

            chunks.append(chunk)
            current_pos = end_char

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        재귀적 텍스트 분할

        Args:
            text: 분할할 텍스트
            separators: 남은 구분자 리스트

        Returns:
            분할된 텍스트 리스트
        """
        # 기저 조건: 텍스트가 충분히 작거나 구분자가 없음
        if not text.strip():
            return []

        text_word_count = len(text.split())
        if text_word_count <= self.chunk_size:
            return [text]

        if not separators:
            # 구분자가 더 이상 없으면 강제로 단어 단위로 분할
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # 현재 구분자로 분할
        parts = self._split_with_separator(text, separator)

        if len(parts) <= 1:
            # 현재 구분자로 분할되지 않으면 다음 구분자로 시도
            return self._split_text(text, remaining_separators)

        # 청크 병합 및 재귀 분할
        chunks = []
        current_chunk_parts = []
        current_word_count = 0

        for part in parts:
            part_word_count = len(part.split())

            # 현재 청크에 추가 가능한지 확인
            if current_word_count + part_word_count <= self.chunk_size:
                current_chunk_parts.append(part)
                current_word_count += part_word_count
            else:
                # 현재 청크 저장
                if current_chunk_parts:
                    merged = self._merge_parts(current_chunk_parts, separator)
                    chunks.append(merged)

                # 파트가 너무 크면 재귀 분할
                if part_word_count > self.chunk_size:
                    sub_chunks = self._split_text(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk_parts = []
                    current_word_count = 0
                else:
                    current_chunk_parts = [part]
                    current_word_count = part_word_count

        # 마지막 청크 저장
        if current_chunk_parts:
            merged = self._merge_parts(current_chunk_parts, separator)
            chunks.append(merged)

        return chunks

    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """구분자로 분할 (구분자 유지 옵션 적용)"""
        if separator == " ":
            # 공백은 단순 분할
            parts = text.split()
            if self.keep_separator:
                return [p + " " for p in parts[:-1]] + [parts[-1]] if parts else []
            return parts

        # 정규식으로 분할 (구분자 유지)
        if self.keep_separator:
            # 구분자를 뒤에 붙여서 유지
            pattern = f"({re.escape(separator)})"
            splits = re.split(pattern, text)

            # 구분자를 이전 파트에 붙이기
            parts = []
            i = 0
            while i < len(splits):
                if i + 1 < len(splits) and splits[i + 1] == separator:
                    parts.append(splits[i] + splits[i + 1])
                    i += 2
                else:
                    if splits[i].strip():
                        parts.append(splits[i])
                    i += 1

            return parts
        else:
            return [p for p in text.split(separator) if p.strip()]

    def _merge_parts(self, parts: List[str], separator: str) -> str:
        """파트들을 구분자로 병합"""
        if self.keep_separator:
            # 구분자가 이미 포함되어 있음
            return "".join(parts)
        else:
            return separator.join(parts)

    def _force_split(self, text: str) -> List[str]:
        """강제 단어 단위 분할 (마지막 수단)"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))

        return chunks
