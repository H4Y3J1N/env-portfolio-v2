"""
Retrieval Evaluator

검색 성능 종합 평가기
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path

from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mrr,
    ndcg_at_k,
    hit_rate_at_k,
    average_precision,
)


@dataclass
class EvaluationResult:
    """단일 쿼리 평가 결과"""
    query_id: str
    query: str
    precision: Dict[int, float] = field(default_factory=dict)  # {k: precision@k}
    recall: Dict[int, float] = field(default_factory=dict)
    f1: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)
    hit_rate: Dict[int, float] = field(default_factory=dict)
    ap: float = 0.0  # Average Precision
    retrieved_chunks: List[str] = field(default_factory=list)
    ground_truth_chunks: List[str] = field(default_factory=list)
    category: str = ""
    difficulty: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "ap": self.ap,
            "retrieved_chunks": self.retrieved_chunks,
            "ground_truth_chunks": self.ground_truth_chunks,
            "category": self.category,
            "difficulty": self.difficulty,
        }


class RetrievalEvaluator:
    """
    검색 성능 평가기

    여러 메트릭을 사용하여 검색 성능을 종합적으로 평가합니다.
    """

    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: 평가할 k 값 리스트 (기본: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        retrieved_chunks: List[str],
        ground_truth: List[str],
        category: str = "",
        difficulty: str = "",
    ) -> EvaluationResult:
        """
        단일 쿼리 평가

        Args:
            query_id: 쿼리 ID
            query: 쿼리 텍스트
            retrieved_chunks: 검색된 청크 ID 리스트 (순위순)
            ground_truth: 정답 청크 ID 리스트
            category: 쿼리 카테고리
            difficulty: 쿼리 난이도

        Returns:
            EvaluationResult
        """
        result = EvaluationResult(
            query_id=query_id,
            query=query,
            retrieved_chunks=retrieved_chunks,
            ground_truth_chunks=ground_truth,
            category=category,
            difficulty=difficulty,
        )

        # 각 k 값에 대해 메트릭 계산
        for k in self.k_values:
            result.precision[k] = precision_at_k(retrieved_chunks, ground_truth, k)
            result.recall[k] = recall_at_k(retrieved_chunks, ground_truth, k)
            result.f1[k] = f1_at_k(retrieved_chunks, ground_truth, k)
            result.ndcg[k] = ndcg_at_k(retrieved_chunks, ground_truth, k)
            result.hit_rate[k] = hit_rate_at_k(retrieved_chunks, ground_truth, k)

        # MRR 및 AP
        result.mrr = mrr(retrieved_chunks, ground_truth)
        result.ap = average_precision(retrieved_chunks, ground_truth)

        return result

    def evaluate_all(
        self,
        test_queries: List[Dict[str, Any]],
        retrieval_results: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        전체 쿼리 평가

        Args:
            test_queries: 테스트 쿼리 리스트
                [{"query_id": str, "query": str, "relevant_chunks": [...], ...}, ...]
            retrieval_results: 검색 결과
                {query_id: [chunk_id, ...], ...}

        Returns:
            종합 평가 결과 딕셔너리
        """
        results = []

        for query_data in test_queries:
            query_id = query_data["query_id"]
            query = query_data.get("query", "")

            # Ground truth 추출
            ground_truth = []
            for chunk_info in query_data.get("relevant_chunks", []):
                if isinstance(chunk_info, str):
                    ground_truth.append(chunk_info)
                elif isinstance(chunk_info, dict):
                    ground_truth.append(chunk_info.get("chunk_id", ""))

            # 검색 결과
            retrieved = retrieval_results.get(query_id, [])

            # 평가
            result = self.evaluate_query(
                query_id=query_id,
                query=query,
                retrieved_chunks=retrieved,
                ground_truth=ground_truth,
                category=query_data.get("category", ""),
                difficulty=query_data.get("difficulty", ""),
            )

            results.append(result)

        # 집계
        aggregated = self._aggregate_results(results)
        aggregated["detailed_results"] = [r.to_dict() for r in results]

        return aggregated

    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """결과 집계"""
        if not results:
            return {
                "num_queries": 0,
                "metrics": {},
            }

        aggregated = {
            "num_queries": len(results),
            "metrics": {},
        }

        # k별 메트릭 집계
        for k in self.k_values:
            aggregated["metrics"][f"precision@{k}"] = np.mean([
                r.precision.get(k, 0) for r in results
            ])
            aggregated["metrics"][f"recall@{k}"] = np.mean([
                r.recall.get(k, 0) for r in results
            ])
            aggregated["metrics"][f"f1@{k}"] = np.mean([
                r.f1.get(k, 0) for r in results
            ])
            aggregated["metrics"][f"ndcg@{k}"] = np.mean([
                r.ndcg.get(k, 0) for r in results
            ])
            aggregated["metrics"][f"hit_rate@{k}"] = np.mean([
                r.hit_rate.get(k, 0) for r in results
            ])

        # MRR, MAP
        aggregated["metrics"]["mrr"] = np.mean([r.mrr for r in results])
        aggregated["metrics"]["map"] = np.mean([r.ap for r in results])

        # 카테고리별 분석
        categories = set(r.category for r in results if r.category)
        if categories:
            aggregated["by_category"] = {}
            for cat in categories:
                cat_results = [r for r in results if r.category == cat]
                aggregated["by_category"][cat] = {
                    "count": len(cat_results),
                    "avg_f1@5": np.mean([r.f1.get(5, 0) for r in cat_results]),
                    "avg_mrr": np.mean([r.mrr for r in cat_results]),
                }

        # 난이도별 분석
        difficulties = set(r.difficulty for r in results if r.difficulty)
        if difficulties:
            aggregated["by_difficulty"] = {}
            for diff in difficulties:
                diff_results = [r for r in results if r.difficulty == diff]
                aggregated["by_difficulty"][diff] = {
                    "count": len(diff_results),
                    "avg_f1@5": np.mean([r.f1.get(5, 0) for r in diff_results]),
                    "avg_mrr": np.mean([r.mrr for r in diff_results]),
                }

        return aggregated

    def save_results(self, results: Dict[str, Any], path: str):
        """결과 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {path}")

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)

        print(f"\nTotal Queries: {results['num_queries']}")

        print("\nMetrics:")
        for metric, value in results.get("metrics", {}).items():
            print(f"  {metric}: {value:.4f}")

        if "by_category" in results:
            print("\nBy Category:")
            for cat, stats in results["by_category"].items():
                print(f"  {cat}: F1@5={stats['avg_f1@5']:.4f}, MRR={stats['avg_mrr']:.4f} (n={stats['count']})")

        if "by_difficulty" in results:
            print("\nBy Difficulty:")
            for diff, stats in results["by_difficulty"].items():
                print(f"  {diff}: F1@5={stats['avg_f1@5']:.4f}, MRR={stats['avg_mrr']:.4f} (n={stats['count']})")

        print("\n" + "=" * 50)
