"""
API Server Tests

API 서버 단위 테스트
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

# 테스트를 위해 main 앱 임포트
import sys
sys.path.insert(0, './api_server')


@pytest.fixture
def mock_inference_client():
    """Mock Inference Client"""
    with patch('services.inference_client.InferenceClient') as mock:
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = {
            "text": "테스트 응답입니다.",
            "tokens_used": 10,
            "finish_reason": "stop",
            "latency_ms": 100.0
        }
        mock_instance.health_check.return_value = {
            "status": "healthy",
            "model": "test-model"
        }
        mock.return_value = mock_instance
        yield mock_instance


class TestHealthEndpoints:
    """Health 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, mock_inference_client):
        """루트 엔드포인트 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_inference_client):
        """헬스 체크 엔드포인트 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "api_server" in data
        assert "inference_server" in data


class TestChatEndpoints:
    """Chat 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_inference_client):
        """채팅 완성 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/completions",
                json={
                    "message": "테스트 메시지",
                    "max_tokens": 100
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "tokens_used" in data

    @pytest.mark.asyncio
    async def test_simple_chat(self, mock_inference_client):
        """간단한 채팅 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/simple",
                json={"message": "안녕하세요"}
            )

        assert response.status_code == 200


class TestToolEndpoints:
    """Tool Calling 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_inference_client):
        """도구 목록 조회 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/tools/available")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_tool_call(self, mock_inference_client):
        """Tool Calling 테스트"""
        # Mock 응답에 tool_call 포함
        mock_inference_client.generate.return_value = {
            "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "서울"}}</tool_call>',
            "tokens_used": 20,
            "finish_reason": "stop",
            "latency_ms": 150.0
        }

        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/tools/call",
                json={
                    "query": "서울 날씨 알려줘",
                    "available_tools": ["get_weather"]
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert "tool_call" in data
        assert data["success"] == True

    @pytest.mark.asyncio
    async def test_tool_execute(self, mock_inference_client):
        """도구 실행 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/tools/execute",
                json={
                    "tool_name": "get_weather",
                    "arguments": {"location": "서울"}
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert data["simulated"] == True
        assert "result" in data


class TestInputValidation:
    """입력 검증 테스트"""

    @pytest.mark.asyncio
    async def test_empty_message(self, mock_inference_client):
        """빈 메시지 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/completions",
                json={"message": "", "max_tokens": 100}
            )

        # 빈 문자열은 Pydantic에서 허용될 수 있음
        # 실제 동작 확인 필요
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, mock_inference_client):
        """잘못된 max_tokens 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/completions",
                json={"message": "테스트", "max_tokens": -1}
            )

        assert response.status_code == 422  # Validation Error

    @pytest.mark.asyncio
    async def test_invalid_temperature(self, mock_inference_client):
        """잘못된 temperature 테스트"""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/completions",
                json={"message": "테스트", "temperature": 3.0}
            )

        assert response.status_code == 422  # Validation Error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
