"""
Weather Tool

날씨 조회 Tool
"""

from typing import Dict, Any, Optional
from datetime import datetime
import random
from .base_tool import BaseTool


class WeatherTool(BaseTool):
    """날씨 조회 Tool"""

    name = "get_weather"
    description = "특정 위치의 현재 또는 예보 날씨 정보를 조회합니다."

    CONDITIONS = ["맑음", "구름조금", "구름많음", "흐림", "비", "눈"]
    CONDITION_WEIGHTS = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]

    def execute(
        self,
        location: str = "서울",
        date: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """날씨 조회

        Args:
            location: 위치
            date: 날짜 (선택)

        Returns:
            날씨 정보
        """
        # 날짜 파싱
        if date:
            try:
                target_date = datetime.fromisoformat(date)
            except ValueError:
                target_date = datetime.now()
        else:
            target_date = datetime.now()

        # 날씨 조건 랜덤 선택 (실제로는 API 호출)
        condition = random.choices(
            self.CONDITIONS,
            weights=self.CONDITION_WEIGHTS,
            k=1,
        )[0]

        # 월별 기준 기온
        monthly_temps = {
            1: (-2, 5), 2: (0, 8), 3: (5, 14), 4: (10, 19),
            5: (15, 24), 6: (20, 28), 7: (24, 31), 8: (25, 32),
            9: (19, 27), 10: (12, 21), 11: (5, 13), 12: (-1, 6),
        }

        month = target_date.month
        min_temp, max_temp = monthly_temps.get(month, (10, 20))
        current_temp = (min_temp + max_temp) / 2 + random.uniform(-3, 3)

        # 강수 확률
        precipitation_prob = {
            "맑음": 0, "구름조금": 10, "구름많음": 20,
            "흐림": 40, "비": 80, "눈": 70,
        }

        # 습도
        humidity = 50 + random.randint(-20, 30)
        if condition in ["비", "눈"]:
            humidity += 20

        return {
            "location": location,
            "date": target_date.strftime("%Y-%m-%d"),
            "condition": condition,
            "temperature": {
                "current": round(current_temp, 1),
                "min": min_temp,
                "max": max_temp,
                "unit": "celsius",
            },
            "humidity_percent": min(100, max(20, humidity)),
            "precipitation_probability": precipitation_prob.get(condition, 20),
            "wind": {
                "speed_ms": round(random.uniform(1, 8), 1),
                "direction": random.choice(["북", "북동", "동", "남동", "남", "남서", "서", "북서"]),
            },
            "uv_index": random.randint(1, 10) if condition == "맑음" else random.randint(1, 4),
        }
