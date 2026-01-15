"""
Evaluation Metrics

검색 성능 평가 메트릭 함수들
"""

from typing import List, Set, Union
import numpy as np


def precision_at_k(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]],
    k: int
) -> float:
    """
    Precision@k 계산

    검색된 상위 k개 중 관련 문서의 비율

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 k개

    Returns:
        Precision@k 값 (0~1)
    """
    if k <= 0:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)

    return relevant_retrieved / k


def recall_at_k(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]],
    k: int
) -> float:
    """
    Recall@k 계산

    전체 관련 문서 중 상위 k개에서 검색된 비율

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 k개

    Returns:
        Recall@k 값 (0~1)
    """
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_set)

    return relevant_retrieved / len(relevant_set)


def f1_at_k(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]],
    k: int
) -> float:
    """
    F1@k 계산

    Precision@k와 Recall@k의 조화 평균

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 k개

    Returns:
        F1@k 값 (0~1)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)


def mrr(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]]
) -> float:
    """
    Mean Reciprocal Rank (MRR) 계산

    첫 번째 관련 문서의 순위의 역수

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합

    Returns:
        MRR 값 (0~1)
    """
    relevant_set = set(relevant)

    for i, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            return 1.0 / i

    return 0.0


def ndcg_at_k(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]],
    k: int,
    relevance_scores: dict = None
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@k) 계산

    순위에 따른 가중 관련성 점수

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 k개
        relevance_scores: 문서별 관련성 점수 (기본: 관련=1, 비관련=0)

    Returns:
        NDCG@k 값 (0~1)
    """
    if k <= 0:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    # 관련성 점수 (기본값: 이진)
    if relevance_scores is None:
        relevance_scores = {doc: 1.0 for doc in relevant_set}

    # DCG 계산
    dcg = 0.0
    for i, doc in enumerate(retrieved_k, 1):
        rel = relevance_scores.get(doc, 0.0)
        dcg += rel / np.log2(i + 1)

    # Ideal DCG (IDCG) 계산
    # 관련 문서를 관련성 점수 순으로 정렬
    sorted_relevances = sorted(
        [relevance_scores.get(doc, 1.0) for doc in relevant_set],
        reverse=True
    )[:k]

    idcg = 0.0
    for i, rel in enumerate(sorted_relevances, 1):
        idcg += rel / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]],
    k: int
) -> float:
    """
    Hit Rate@k 계산

    상위 k개에 관련 문서가 하나라도 있으면 1, 없으면 0

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합
        k: 상위 k개

    Returns:
        Hit Rate (0 또는 1)
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return 1.0 if retrieved_k & relevant_set else 0.0


def average_precision(
    retrieved: List[str],
    relevant: Union[List[str], Set[str]]
) -> float:
    """
    Average Precision (AP) 계산

    각 관련 문서 위치에서의 Precision의 평균

    Args:
        retrieved: 검색된 문서 ID 리스트 (순위순)
        relevant: 관련 문서 ID 집합

    Returns:
        AP 값 (0~1)
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    relevant_count = 0
    precision_sum = 0.0

    for i, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            relevant_count += 1
            precision_sum += relevant_count / i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant_set)
