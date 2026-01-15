# Chunkers Module

from .base_chunker import BaseChunker, Chunk
from .fixed_chunker import FixedChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
]


def get_chunker(strategy: str, config: dict) -> BaseChunker:
    """
    전략에 따른 청커 인스턴스 생성

    Args:
        strategy: 청킹 전략 (fixed, recursive, semantic)
        config: 청킹 설정

    Returns:
        BaseChunker 인스턴스
    """
    chunkers = {
        "fixed": FixedChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(chunkers.keys())}")

    return chunkers[strategy](config)
