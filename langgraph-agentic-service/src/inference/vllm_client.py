"""
vLLM Client

vLLM 서버와 통신하는 클라이언트
"""

import logging
import os
from typing import Optional, Dict, Any, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM 서버 클라이언트

    vLLM OpenAI-compatible API와 통신
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Args:
            base_url: vLLM 서버 URL
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url or os.getenv("VLLM_SERVER_URL", "http://localhost:8001")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        lora_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 temperature
            top_p: Top-p sampling
            stop: 정지 토큰 리스트
            lora_name: LoRA adapter 이름 (멀티테넌트용)

        Returns:
            생성 결과
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop or [],
                }

                # LoRA adapter 지정
                if lora_name:
                    request_body["lora_request"] = {"lora_name": lora_name}

                response = await client.post(
                    f"{self.base_url}/v1/completions",
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()
                return {
                    "text": result["choices"][0]["text"],
                    "usage": result.get("usage", {}),
                    "finish_reason": result["choices"][0].get("finish_reason"),
                }

            except httpx.TimeoutException:
                logger.error("vLLM request timeout")
                raise Exception("Request timeout")

            except httpx.HTTPStatusError as e:
                logger.error(f"vLLM HTTP error: {e}")
                raise Exception(f"Server error: {e.response.status_code}")

            except Exception as e:
                logger.error(f"vLLM request error: {e}")
                raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        lora_name: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 temperature
            lora_name: LoRA adapter 이름

        Yields:
            생성된 토큰
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }

                if lora_name:
                    request_body["lora_request"] = {"lora_name": lora_name}

                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    json=request_body,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break

                            import json
                            chunk = json.loads(data)
                            token = chunk["choices"][0].get("text", "")
                            if token:
                                yield token

            except Exception as e:
                logger.error(f"vLLM stream error: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """헬스체크

        Returns:
            서버 상태 정보
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"vLLM health check failed: {e}")
                return {"status": "unavailable", "error": str(e)}

    async def get_models(self) -> list[str]:
        """사용 가능한 모델 목록

        Returns:
            모델 이름 리스트
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
            except Exception as e:
                logger.error(f"Failed to get models: {e}")
                return []
