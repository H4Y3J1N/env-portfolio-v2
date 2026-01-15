"""
Load Test with Locust

부하 테스트 스크립트 (Locust 프레임워크)

실행 방법:
    locust -f benchmarks/load_test.py --host http://localhost:8000

웹 UI:
    http://localhost:8089
"""

import json
import random
from locust import HttpUser, task, between, events


# ===========================================
# Test Data
# ===========================================
CHAT_MESSAGES = [
    "내일 오전 10시 기온을 예측해주세요.",
    "우리 농장의 토마토 수확량을 예측해줘.",
    "패널 각도를 최적화하려면 어떻게 해야 하나요?",
    "다음 주 날씨는 어떨까요?",
    "상추 200평 재배하면 수확량이 얼마나 될까요?",
    "서울 지역 이번 달 태양광 패널 최적 각도는?",
    "고추 재배에 적합한 온도는 몇 도인가요?",
    "오늘 부산 날씨 알려주세요.",
]

TOOL_QUERIES = [
    {"query": "서울 내일 날씨 알려줘", "tools": ["get_weather"]},
    {"query": "토마토 100평 수확량 예측해줘", "tools": ["predict_yield"]},
    {"query": "위도 37도에서 6월 태양광 패널 각도", "tools": ["optimize_panel_angle"]},
    {"query": "대전 다음주 월요일 기온 예측", "tools": ["predict_temperature"]},
    {"query": "상추 300평 재배 시 예상 수확량", "tools": ["predict_yield"]},
]


# ===========================================
# Locust User Class
# ===========================================
class APIUser(HttpUser):
    """API 사용자 시뮬레이션"""

    # 요청 간 대기 시간 (초)
    wait_time = between(1, 3)

    def on_start(self):
        """테스트 시작 시 실행"""
        # 헬스 체크
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Warning: Health check failed with status {response.status_code}")

    @task(5)
    def chat_completion(self):
        """채팅 완성 요청 (50% 비중)"""
        message = random.choice(CHAT_MESSAGES)

        with self.client.post(
            "/api/v1/chat/completions",
            json={
                "message": message,
                "max_tokens": 256,
                "temperature": 0.7
            },
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def tool_calling(self):
        """Tool Calling 요청 (30% 비중)"""
        test_case = random.choice(TOOL_QUERIES)

        with self.client.post(
            "/api/v1/tools/call",
            json={
                "query": test_case["query"],
                "available_tools": test_case["tools"],
                "max_tokens": 256,
                "temperature": 0.3
            },
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    # Tool Call 추출 실패도 일단 성공으로 처리 (서버는 정상 동작)
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """헬스 체크 (10% 비중)"""
        self.client.get("/health")

    @task(1)
    def list_tools(self):
        """도구 목록 조회 (10% 비중)"""
        self.client.get("/api/v1/tools/available")


# ===========================================
# Event Handlers
# ===========================================
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """테스트 시작 이벤트"""
    print("=" * 60)
    print("Load Test Starting...")
    print(f"Target Host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """테스트 종료 이벤트"""
    print("=" * 60)
    print("Load Test Completed!")
    print("=" * 60)


# ===========================================
# Custom Command Line Options
# ===========================================
@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    """커스텀 명령줄 옵션 추가"""
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for generation"
    )


# ===========================================
# Main (for testing)
# ===========================================
if __name__ == "__main__":
    print("""
vLLM Load Test Script

Usage:
    locust -f benchmarks/load_test.py --host http://localhost:8000

Options:
    --users        : Number of concurrent users
    --spawn-rate   : Users spawned per second
    --run-time     : Test duration (e.g., 60s, 5m)
    --headless     : Run without web UI

Example:
    locust -f benchmarks/load_test.py --host http://localhost:8000 \\
           --users 10 --spawn-rate 2 --run-time 60s --headless
    """)
