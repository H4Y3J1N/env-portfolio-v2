"""
Temperature Predictor Tool

기온 예측 Tool
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import random
from .base_tool import BaseTool


class TemperaturePredictorTool(BaseTool):
    """기온 예측 Tool"""

    name = "predict_temperature"
    description = "특정 날짜와 시간의 예상 기온을 예측합니다."

    def execute(
        self,
        datetime_str: Optional[str] = None,
        location: str = "서울",
        **kwargs,
    ) -> Dict[str, Any]:
        """기온 예측

        Args:
            datetime_str: 예측 일시 (ISO 형식)
            location: 위치

        Returns:
            예측 결과
        """
        # 날짜 파싱
        if datetime_str:
            try:
                target_dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
            except ValueError:
                target_dt = datetime.now() + timedelta(days=1)
        else:
            target_dt = datetime.now() + timedelta(days=1)

        # 월별 기준 기온 (서울 기준)
        monthly_temps = {
            1: (-2, 5), 2: (0, 8), 3: (5, 14), 4: (10, 19),
            5: (15, 24), 6: (20, 28), 7: (24, 31), 8: (25, 32),
            9: (19, 27), 10: (12, 21), 11: (5, 13), 12: (-1, 6),
        }

        month = target_dt.month
        min_temp, max_temp = monthly_temps.get(month, (10, 20))

        # 시간대별 조정
        hour = target_dt.hour
        if 6 <= hour < 10:
            temp = min_temp + (max_temp - min_temp) * 0.3
        elif 10 <= hour < 14:
            temp = min_temp + (max_temp - min_temp) * 0.8
        elif 14 <= hour < 18:
            temp = max_temp - (max_temp - min_temp) * 0.1
        else:
            temp = min_temp + (max_temp - min_temp) * 0.2

        # 약간의 랜덤성 추가
        temp += random.uniform(-2, 2)

        return {
            "datetime": target_dt.isoformat(),
            "location": location,
            "predicted_temp_celsius": round(temp, 1),
            "min_temp_celsius": min_temp,
            "max_temp_celsius": max_temp,
            "confidence": 0.82,
            "unit": "celsius",
        }
