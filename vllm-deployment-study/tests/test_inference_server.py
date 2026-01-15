"""
Inference Server Tests

vLLM Inference Server 단위 테스트
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
import sys

sys.path.insert(0, './inference_server')


@pytest.fixture
def mock_vllm_engine():
    """Mock vLLM Engine"""
    with patch('vllm.AsyncLLMEngine') as mock:
        mock_instance = AsyncMock()

        # Mock generate response
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "테스트 응답입니다."
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2, 3, 4, 5]
        mock_output.prompt_token_ids = [1, 2, 3]

        async def mock_generate(*args, **kwargs):
            yield mock_output

        mock_instance.generate = mock_generate
        mock.from_engine_args.return_value = mock_instance

        yield mock_instance


class TestHealthEndpoint:
    """Health 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_vllm_engine):
        """정상 상태 헬스 체크 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "gpu_memory" in data

    @pytest.mark.asyncio
    async def test_health_response_format(self, mock_vllm_engine):
        """헬스 응답 포맷 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")

        data = response.json()
        required_fields = ["status", "model", "gpu_memory", "timestamp"]
        for field in required_fields:
            assert field in data


class TestGenerateEndpoint:
    """Generate 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_vllm_engine):
        """정상 생성 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={
                    "prompt": "테스트 프롬프트",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "tokens_used" in data
        assert "finish_reason" in data

    @pytest.mark.asyncio
    async def test_generate_with_default_params(self, mock_vllm_engine):
        """기본 파라미터 생성 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={"prompt": "안녕하세요"}
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, mock_vllm_engine):
        """빈 프롬프트 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={"prompt": ""}
            )

        # 빈 프롬프트는 거부되어야 함
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_generate_invalid_temperature(self, mock_vllm_engine):
        """잘못된 temperature 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={
                    "prompt": "테스트",
                    "temperature": 5.0  # 범위 초과
                }
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_generate_invalid_max_tokens(self, mock_vllm_engine):
        """잘못된 max_tokens 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={
                    "prompt": "테스트",
                    "max_tokens": -10
                }
            )

        assert response.status_code == 422


class TestStreamingEndpoint:
    """스트리밍 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_streaming_generate(self, mock_vllm_engine):
        """스트리밍 생성 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate/stream",
                json={
                    "prompt": "테스트 프롬프트",
                    "max_tokens": 50
                }
            )

        assert response.status_code == 200
        # 스트리밍 응답은 text/event-stream 타입
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestMetricsEndpoint:
    """메트릭 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, mock_vllm_engine):
        """메트릭 엔드포인트 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        # 필수 메트릭 필드 확인
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_latency_ms" in data


class TestInputValidation:
    """입력 검증 테스트"""

    @pytest.mark.asyncio
    async def test_missing_prompt(self, mock_vllm_engine):
        """프롬프트 누락 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                json={"max_tokens": 100}
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_vllm_engine):
        """잘못된 JSON 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/generate",
                content="invalid json",
                headers={"Content-Type": "application/json"}
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_top_p_range(self, mock_vllm_engine):
        """top_p 범위 테스트"""
        from vllm_server import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            # top_p > 1.0
            response = await client.post(
                "/generate",
                json={
                    "prompt": "테스트",
                    "top_p": 1.5
                }
            )

        assert response.status_code == 422


class TestConcurrency:
    """동시성 테스트"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_vllm_engine):
        """동시 요청 처리 테스트"""
        import asyncio
        from vllm_server import app

        async def make_request(client, idx):
            response = await client.post(
                "/generate",
                json={
                    "prompt": f"테스트 요청 {idx}",
                    "max_tokens": 10
                }
            )
            return response.status_code

        async with AsyncClient(app=app, base_url="http://test") as client:
            tasks = [make_request(client, i) for i in range(5)]
            results = await asyncio.gather(*tasks)

        # 모든 요청이 성공해야 함
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
