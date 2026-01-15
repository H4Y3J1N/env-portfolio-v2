"""
Pytest Configuration

공유 픽스처 및 설정
"""

import pytest
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "api_server"))
sys.path.insert(0, str(project_root / "inference_server"))


@pytest.fixture(scope="session")
def event_loop():
    """세션 범위 이벤트 루프"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_prompts():
    """테스트용 샘플 프롬프트"""
    return [
        "안녕하세요!",
        "오늘 날씨가 어떻습니까?",
        "서울의 날씨를 알려주세요.",
        "농작물 수확량을 예측해주세요.",
        "태양광 패널의 최적 각도는 무엇인가요?",
    ]


@pytest.fixture
def sample_tool_queries():
    """Tool Calling 테스트용 쿼리"""
    return [
        {
            "query": "서울 날씨 알려줘",
            "expected_tool": "get_weather"
        },
        {
            "query": "내일 사과 수확량 예측해줘",
            "expected_tool": "predict_yield"
        },
        {
            "query": "태양광 패널 각도 최적화해줘",
            "expected_tool": "optimize_panel_angle"
        },
    ]


@pytest.fixture
def mock_generate_response():
    """Mock 생성 응답"""
    return {
        "text": "테스트 응답입니다. 이것은 vLLM 추론 서버의 모의 응답입니다.",
        "tokens_used": 25,
        "finish_reason": "stop",
        "latency_ms": 150.0
    }


@pytest.fixture
def mock_health_response():
    """Mock 헬스 체크 응답"""
    return {
        "status": "healthy",
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpu_memory": {
            "used": "10.5 GB",
            "total": "24.0 GB",
            "utilization": "43.75%"
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def api_server_config():
    """API 서버 테스트 설정"""
    return {
        "base_url": "http://test",
        "timeout": 30.0,
        "max_retries": 3
    }


@pytest.fixture
def inference_server_config():
    """추론 서버 테스트 설정"""
    return {
        "base_url": "http://test",
        "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "max_tokens": 512,
        "temperature": 0.7
    }
