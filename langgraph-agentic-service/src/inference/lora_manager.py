"""
LoRA Manager

멀티테넌트 LoRA Adapter 동적 로딩 관리
"""

import logging
import os
from typing import Dict, Optional, Any
import httpx

logger = logging.getLogger(__name__)


class LoRAManager:
    """멀티테넌트 LoRA Adapter 관리

    테넌트별 LoRA adapter 동적 로딩 및 추론 지원
    """

    def __init__(
        self,
        vllm_server_url: Optional[str] = None,
    ):
        """
        Args:
            vllm_server_url: vLLM 서버 URL
        """
        self.vllm_url = vllm_server_url or os.getenv(
            "VLLM_SERVER_URL", "http://localhost:8001"
        )

        # 로드된 adapter 캐시
        self.loaded_adapters: Dict[str, str] = {}

        # 테넌트별 LoRA 매핑
        self.tenant_adapters = {
            "korea": {
                "path": "adapters/korea_agriculture_lora",
                "description": "한국 농업 도메인 특화 LoRA",
            },
            "japan": {
                "path": "adapters/japan_agriculture_lora",
                "description": "일본 농업 도메인 특화 LoRA",
            },
            "vietnam": {
                "path": "adapters/vietnam_agriculture_lora",
                "description": "베트남 농업 도메인 특화 LoRA",
            },
            "thailand": {
                "path": "adapters/thailand_agriculture_lora",
                "description": "태국 농업 도메인 특화 LoRA",
            },
        }

    def get_adapter_for_tenant(self, tenant_id: str) -> Optional[Dict[str, str]]:
        """테넌트에 해당하는 LoRA adapter 정보

        Args:
            tenant_id: 테넌트 ID

        Returns:
            adapter 정보 또는 None
        """
        return self.tenant_adapters.get(tenant_id)

    async def ensure_adapter_loaded(self, tenant_id: str) -> bool:
        """LoRA adapter 로딩 확인/로드

        Args:
            tenant_id: 테넌트 ID

        Returns:
            로딩 성공 여부
        """
        adapter_info = self.get_adapter_for_tenant(tenant_id)

        if adapter_info is None:
            logger.debug(f"No adapter configured for tenant: {tenant_id}")
            return True  # 기본 모델 사용

        if tenant_id in self.loaded_adapters:
            logger.debug(f"Adapter already loaded: {tenant_id}")
            return True

        # vLLM 서버에 adapter 로딩 요청
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.vllm_url}/v1/load_lora_adapter",
                    json={
                        "lora_name": tenant_id,
                        "lora_path": adapter_info["path"],
                    },
                )

                if response.status_code == 200:
                    self.loaded_adapters[tenant_id] = adapter_info["path"]
                    logger.info(f"LoRA adapter loaded: {tenant_id}")
                    return True
                else:
                    logger.error(f"Failed to load adapter: {response.text}")
                    return False

            except httpx.HTTPStatusError as e:
                # 404는 엔드포인트 없음 (vLLM 버전에 따라 다름)
                if e.response.status_code == 404:
                    logger.warning("LoRA loading endpoint not available")
                    return True
                logger.error(f"HTTP error loading adapter: {e}")
                return False

            except Exception as e:
                logger.error(f"Error loading adapter: {e}")
                return False

    async def generate_with_adapter(
        self,
        prompt: str,
        tenant_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """테넌트별 LoRA adapter로 추론

        Args:
            prompt: 입력 프롬프트
            tenant_id: 테넌트 ID
            max_tokens: 최대 토큰 수
            temperature: 생성 temperature
            **kwargs: 추가 생성 파라미터

        Returns:
            생성 결과
        """
        # Adapter 로딩 확인
        await self.ensure_adapter_loaded(tenant_id)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                }

                # LoRA adapter 지정
                if tenant_id in self.loaded_adapters:
                    request_body["lora_request"] = {"lora_name": tenant_id}

                response = await client.post(
                    f"{self.vllm_url}/v1/completions",
                    json=request_body,
                )
                response.raise_for_status()

                result = response.json()
                return {
                    "text": result["choices"][0]["text"],
                    "usage": result.get("usage", {}),
                    "tenant_id": tenant_id,
                    "adapter_used": tenant_id in self.loaded_adapters,
                }

            except Exception as e:
                logger.error(f"Generation with adapter error: {e}")
                raise

    def list_available_adapters(self) -> Dict[str, Dict]:
        """사용 가능한 adapter 목록

        Returns:
            테넌트별 adapter 정보
        """
        return self.tenant_adapters.copy()

    def list_loaded_adapters(self) -> Dict[str, str]:
        """현재 로드된 adapter 목록

        Returns:
            로드된 adapter 딕셔너리
        """
        return self.loaded_adapters.copy()

    async def unload_adapter(self, tenant_id: str) -> bool:
        """Adapter 언로드

        Args:
            tenant_id: 테넌트 ID

        Returns:
            언로드 성공 여부
        """
        if tenant_id not in self.loaded_adapters:
            return True

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.vllm_url}/v1/unload_lora_adapter",
                    json={"lora_name": tenant_id},
                )

                if response.status_code == 200:
                    del self.loaded_adapters[tenant_id]
                    logger.info(f"LoRA adapter unloaded: {tenant_id}")
                    return True

                return False

            except Exception as e:
                logger.error(f"Error unloading adapter: {e}")
                return False


class MockLoRAManager(LoRAManager):
    """테스트용 Mock LoRA Manager"""

    async def ensure_adapter_loaded(self, tenant_id: str) -> bool:
        """Mock 로딩"""
        adapter_info = self.get_adapter_for_tenant(tenant_id)
        if adapter_info:
            self.loaded_adapters[tenant_id] = adapter_info["path"]
        return True

    async def generate_with_adapter(
        self,
        prompt: str,
        tenant_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Mock 생성"""
        await self.ensure_adapter_loaded(tenant_id)

        return {
            "text": f"[{tenant_id} LoRA] Mock response for: {prompt[:50]}...",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "tenant_id": tenant_id,
            "adapter_used": tenant_id in self.loaded_adapters,
        }
