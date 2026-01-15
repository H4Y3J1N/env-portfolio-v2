"""
Inference Server Client

vLLM 추론 서버와 통신하는 HTTP 클라이언트
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class InferenceClient:
    """vLLM Inference 서버 클라이언트"""

    def __init__(self):
        self.base_url = os.getenv("INFERENCE_SERVER_URL", "http://inference-server:8001")
        self.timeout = float(os.getenv("INFERENCE_TIMEOUT", "120.0"))
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        logger.debug(f"InferenceClient initialized: {self.base_url}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        텍스트 생성 요청

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링
            top_k: Top-k 샘플링
            repetition_penalty: 반복 패널티
            stop: 중단 토큰 목록

        Returns:
            생성 결과 딕셔너리
        """
        # Mock 모드
        if self.mock_mode:
            logger.info("Running in MOCK mode")
            return {
                "text": "[MOCK] 이것은 테스트 응답입니다. 실제 모델이 연결되지 않았습니다.",
                "tokens_used": 15,
                "finish_reason": "mock",
                "latency_ms": 50.0
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.debug(f"Sending generate request to {self.base_url}")

                response = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "stop": stop,
                        "stream": False
                    }
                )

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                logger.error("Inference request timeout")
                raise InferenceError("Request timeout - inference server may be overloaded")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code}")
                detail = e.response.text if e.response.text else str(e)
                raise InferenceError(f"Inference server error ({e.response.status_code}): {detail}")

            except httpx.ConnectError:
                logger.error("Failed to connect to inference server")
                raise InferenceError("Cannot connect to inference server")

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise InferenceError(f"Unexpected error: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """
        헬스 체크

        Returns:
            서버 상태 정보
        """
        # Mock 모드
        if self.mock_mode:
            return {
                "status": "mock",
                "model": "mock-model",
                "quantization": "none"
            }

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                raise InferenceError("Health check timeout")

            except httpx.ConnectError:
                raise InferenceError("Cannot connect to inference server")

            except Exception as e:
                raise InferenceError(f"Health check failed: {str(e)}")

    async def get_metrics(self) -> Dict[str, Any]:
        """
        메트릭 조회

        Returns:
            서버 메트릭 정보
        """
        if self.mock_mode:
            return {"mock": True}

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/metrics")
                response.raise_for_status()
                return response.json()

            except Exception as e:
                raise InferenceError(f"Failed to get metrics: {str(e)}")


class InferenceError(Exception):
    """추론 서버 에러"""
    pass
