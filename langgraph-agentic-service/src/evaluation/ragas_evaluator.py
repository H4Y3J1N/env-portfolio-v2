"""
RAGAS Evaluator

RAG 품질 평가를 위한 RAGAS 메트릭 구현
- Faithfulness: 답변이 컨텍스트에 충실한지
- Answer Relevancy: 답변이 질문과 관련있는지
- Context Precision: 검색된 컨텍스트가 정확한지
- Context Recall: 필요한 정보가 모두 검색되었는지
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """평가 결과"""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # RAGAS 메트릭
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # 종합 점수
    overall_score: float = 0.0

    # 메타데이터
    evaluation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metrics": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "overall_score": self.overall_score,
            },
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp,
        }


class RAGASEvaluator:
    """RAGAS 평가기

    LLM 기반 RAG 품질 평가
    - 실제 운영에서는 OpenAI/Anthropic API 또는 로컬 LLM 사용
    - 여기서는 규칙 기반 + LLM 호출 시뮬레이션으로 구현
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_mock: bool = True,
    ):
        """
        Args:
            llm_client: LLM 클라이언트 (vLLM, OpenAI 등)
            use_mock: Mock 모드 사용 여부
        """
        self.llm_client = llm_client
        self.use_mock = use_mock or llm_client is None

        logger.info(f"RAGASEvaluator initialized (mock={self.use_mock})")

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """단일 샘플 평가

        Args:
            question: 질문
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 목록
            ground_truth: 정답 (선택)

        Returns:
            평가 결과
        """
        import time
        start_time = time.time()

        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        # 각 메트릭 계산
        if self.use_mock:
            # Mock 평가 (규칙 기반)
            result.faithfulness = await self._calculate_faithfulness_mock(answer, contexts)
            result.answer_relevancy = await self._calculate_answer_relevancy_mock(question, answer)
            result.context_precision = await self._calculate_context_precision_mock(question, contexts)
            result.context_recall = await self._calculate_context_recall_mock(contexts, ground_truth)
        else:
            # LLM 기반 평가
            result.faithfulness = await self._calculate_faithfulness_llm(answer, contexts)
            result.answer_relevancy = await self._calculate_answer_relevancy_llm(question, answer)
            result.context_precision = await self._calculate_context_precision_llm(question, contexts)
            result.context_recall = await self._calculate_context_recall_llm(contexts, ground_truth)

        # 종합 점수 (가중 평균)
        result.overall_score = (
            result.faithfulness * 0.3 +
            result.answer_relevancy * 0.3 +
            result.context_precision * 0.2 +
            result.context_recall * 0.2
        )

        result.evaluation_time = time.time() - start_time

        return result

    async def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        concurrency: int = 5,
    ) -> List[EvaluationResult]:
        """배치 평가

        Args:
            samples: 평가 샘플 목록
                [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}, ...]
            concurrency: 동시 처리 수

        Returns:
            평가 결과 목록
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_with_semaphore(sample: Dict[str, Any]) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(
                    question=sample["question"],
                    answer=sample["answer"],
                    contexts=sample.get("contexts", []),
                    ground_truth=sample.get("ground_truth"),
                )

        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        results = await asyncio.gather(*tasks)

        return list(results)

    def generate_report(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """평가 리포트 생성

        Args:
            results: 평가 결과 목록

        Returns:
            종합 리포트
        """
        if not results:
            return {"error": "No results to report"}

        # 메트릭별 평균 계산
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_answer_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_context_precision = sum(r.context_precision for r in results) / len(results)
        avg_context_recall = sum(r.context_recall for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)

        # 메트릭별 분포
        def get_distribution(values: List[float]) -> Dict[str, int]:
            distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
            for v in values:
                if v >= 0.8:
                    distribution["excellent"] += 1
                elif v >= 0.6:
                    distribution["good"] += 1
                elif v >= 0.4:
                    distribution["fair"] += 1
                else:
                    distribution["poor"] += 1
            return distribution

        report = {
            "summary": {
                "total_samples": len(results),
                "average_scores": {
                    "faithfulness": round(avg_faithfulness, 4),
                    "answer_relevancy": round(avg_answer_relevancy, 4),
                    "context_precision": round(avg_context_precision, 4),
                    "context_recall": round(avg_context_recall, 4),
                    "overall": round(avg_overall, 4),
                },
                "evaluation_time_total": round(sum(r.evaluation_time for r in results), 2),
            },
            "distribution": {
                "overall": get_distribution([r.overall_score for r in results]),
                "faithfulness": get_distribution([r.faithfulness for r in results]),
                "answer_relevancy": get_distribution([r.answer_relevancy for r in results]),
            },
            "recommendations": self._generate_recommendations(
                avg_faithfulness, avg_answer_relevancy,
                avg_context_precision, avg_context_recall
            ),
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _generate_recommendations(
        self,
        faithfulness: float,
        answer_relevancy: float,
        context_precision: float,
        context_recall: float,
    ) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        if faithfulness < 0.7:
            recommendations.append(
                "Faithfulness가 낮습니다. 답변 생성 시 컨텍스트를 더 강하게 참조하도록 "
                "프롬프트를 개선하거나, 환각을 줄이기 위한 후처리를 추가하세요."
            )

        if answer_relevancy < 0.7:
            recommendations.append(
                "Answer Relevancy가 낮습니다. 질문의 의도를 더 정확히 파악하도록 "
                "프롬프트 엔지니어링을 개선하세요."
            )

        if context_precision < 0.7:
            recommendations.append(
                "Context Precision이 낮습니다. Retriever의 정밀도를 높이기 위해 "
                "Reranker 가중치 조정 또는 top_k 값을 낮추는 것을 고려하세요."
            )

        if context_recall < 0.7:
            recommendations.append(
                "Context Recall이 낮습니다. 더 많은 관련 문서를 검색하도록 "
                "top_k를 높이거나 Hybrid Search 비중을 조정하세요."
            )

        if not recommendations:
            recommendations.append(
                "전반적으로 좋은 성능을 보이고 있습니다. "
                "지속적인 모니터링과 A/B 테스트를 통해 최적화를 유지하세요."
            )

        return recommendations

    # ==================== Mock 평가 메서드 ====================

    async def _calculate_faithfulness_mock(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Faithfulness Mock 계산

        답변의 각 문장이 컨텍스트에서 지원되는지 확인
        """
        if not contexts or not answer:
            return 0.5

        # 컨텍스트 합치기
        context_text = " ".join(contexts).lower()

        # 답변을 문장으로 분리
        sentences = [s.strip() for s in answer.replace(".", ".|||").split("|||") if s.strip()]
        if not sentences:
            return 0.5

        # 각 문장에서 키워드 추출 후 컨텍스트와 매칭
        supported_count = 0
        for sentence in sentences:
            words = sentence.lower().split()
            # 3글자 이상 단어만 체크
            keywords = [w for w in words if len(w) >= 3]
            if not keywords:
                continue

            # 키워드의 50% 이상이 컨텍스트에 있으면 지원됨
            matched = sum(1 for kw in keywords if kw in context_text)
            if matched / len(keywords) >= 0.3:
                supported_count += 1

        score = supported_count / len(sentences) if sentences else 0.5
        return min(1.0, max(0.0, score + 0.1))  # 약간의 보정

    async def _calculate_answer_relevancy_mock(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Answer Relevancy Mock 계산

        질문과 답변의 키워드 오버랩 체크
        """
        if not question or not answer:
            return 0.5

        # 질문 키워드 추출
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # 불용어 제거 (간단한 한국어/영어 불용어)
        stopwords = {
            "은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로",
            "와", "과", "도", "만", "부터", "까지", "라고", "라는", "하는", "있는",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "what", "how", "when", "where", "why", "which", "who",
        }

        question_keywords = question_words - stopwords
        answer_keywords = answer_words - stopwords

        if not question_keywords:
            return 0.7

        # 오버랩 비율
        overlap = len(question_keywords & answer_keywords)
        relevancy = overlap / len(question_keywords)

        # 답변 길이 보정 (너무 짧거나 길면 감점)
        length_ratio = len(answer) / max(len(question), 1)
        if length_ratio < 0.5:
            relevancy *= 0.8
        elif length_ratio > 10:
            relevancy *= 0.9

        return min(1.0, max(0.0, relevancy + 0.2))

    async def _calculate_context_precision_mock(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        """Context Precision Mock 계산

        검색된 컨텍스트가 질문과 관련 있는지 체크
        """
        if not contexts or not question:
            return 0.5

        question_words = set(question.lower().split())

        relevant_count = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(question_words & ctx_words)
            if overlap >= 2:  # 최소 2개 키워드 매칭
                relevant_count += 1

        precision = relevant_count / len(contexts)
        return min(1.0, max(0.0, precision + 0.15))

    async def _calculate_context_recall_mock(
        self,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> float:
        """Context Recall Mock 계산

        정답에 필요한 정보가 컨텍스트에 있는지 체크
        """
        if not ground_truth:
            # Ground truth 없으면 컨텍스트 양으로 추정
            if len(contexts) >= 3:
                return 0.8
            elif len(contexts) >= 1:
                return 0.6
            return 0.4

        context_text = " ".join(contexts).lower()
        ground_truth_words = set(ground_truth.lower().split())

        # Ground truth 키워드가 컨텍스트에 포함된 비율
        found = sum(1 for w in ground_truth_words if w in context_text)
        recall = found / len(ground_truth_words) if ground_truth_words else 0.5

        return min(1.0, max(0.0, recall + 0.1))

    # ==================== LLM 기반 평가 메서드 ====================

    async def _calculate_faithfulness_llm(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Faithfulness LLM 기반 계산"""
        if not self.llm_client:
            return await self._calculate_faithfulness_mock(answer, contexts)

        prompt = f"""다음 답변이 주어진 컨텍스트에 얼마나 충실한지 0.0~1.0 사이의 점수로 평가해주세요.
답변에 포함된 정보가 컨텍스트에서 직접 지원되는지 확인하세요.

컨텍스트:
{chr(10).join(contexts)}

답변:
{answer}

점수만 숫자로 응답해주세요 (예: 0.85):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"LLM faithfulness evaluation failed: {e}")
            return await self._calculate_faithfulness_mock(answer, contexts)

    async def _calculate_answer_relevancy_llm(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Answer Relevancy LLM 기반 계산"""
        if not self.llm_client:
            return await self._calculate_answer_relevancy_mock(question, answer)

        prompt = f"""다음 답변이 질문에 얼마나 관련 있는지 0.0~1.0 사이의 점수로 평가해주세요.
답변이 질문의 의도를 잘 파악하고 적절히 응답했는지 확인하세요.

질문:
{question}

답변:
{answer}

점수만 숫자로 응답해주세요 (예: 0.85):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"LLM answer relevancy evaluation failed: {e}")
            return await self._calculate_answer_relevancy_mock(question, answer)

    async def _calculate_context_precision_llm(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        """Context Precision LLM 기반 계산"""
        if not self.llm_client:
            return await self._calculate_context_precision_mock(question, contexts)

        prompt = f"""다음 검색된 컨텍스트들이 질문에 얼마나 정확하게 관련되어 있는지 0.0~1.0 사이의 점수로 평가해주세요.

질문:
{question}

검색된 컨텍스트:
{chr(10).join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])}

점수만 숫자로 응답해주세요 (예: 0.85):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"LLM context precision evaluation failed: {e}")
            return await self._calculate_context_precision_mock(question, contexts)

    async def _calculate_context_recall_llm(
        self,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> float:
        """Context Recall LLM 기반 계산"""
        if not self.llm_client or not ground_truth:
            return await self._calculate_context_recall_mock(contexts, ground_truth)

        prompt = f"""주어진 정답을 생성하기 위해 필요한 정보가 검색된 컨텍스트에 얼마나 포함되어 있는지
0.0~1.0 사이의 점수로 평가해주세요.

정답:
{ground_truth}

검색된 컨텍스트:
{chr(10).join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])}

점수만 숫자로 응답해주세요 (예: 0.85):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=10)
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"LLM context recall evaluation failed: {e}")
            return await self._calculate_context_recall_mock(contexts, ground_truth)


async def run_evaluation_example():
    """평가 예제 실행"""
    evaluator = RAGASEvaluator(use_mock=True)

    # 테스트 샘플
    samples = [
        {
            "question": "토마토 재배 시 적정 온도는 얼마인가요?",
            "answer": "토마토는 낮 기온 25-30도, 밤 기온 15-20도에서 가장 잘 자랍니다. "
                     "너무 높거나 낮은 온도는 생육에 악영향을 줍니다.",
            "contexts": [
                "토마토의 적정 생육 온도는 낮에 25-30도, 밤에 15-20도입니다.",
                "온도가 35도를 넘으면 화분 발아율이 떨어집니다.",
                "10도 이하에서는 생육이 정지됩니다.",
            ],
            "ground_truth": "토마토 적정 온도: 낮 25-30도, 밤 15-20도",
        },
        {
            "question": "태양광 패널 최적 각도는?",
            "answer": "태양광 패널의 최적 각도는 위도와 계절에 따라 다릅니다. "
                     "한국의 경우 연평균 30-35도가 적합합니다.",
            "contexts": [
                "한국에서 태양광 패널 최적 각도는 위도에서 약간 빼서 30-35도입니다.",
                "여름에는 각도를 낮추고 겨울에는 높이는 것이 효율적입니다.",
            ],
            "ground_truth": "한국 태양광 패널 최적 각도: 30-35도",
        },
    ]

    # 배치 평가
    results = await evaluator.evaluate_batch(samples)

    # 리포트 생성
    report = evaluator.generate_report(results)

    print("=" * 50)
    print("RAGAS Evaluation Report")
    print("=" * 50)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return results, report


if __name__ == "__main__":
    asyncio.run(run_evaluation_example())
