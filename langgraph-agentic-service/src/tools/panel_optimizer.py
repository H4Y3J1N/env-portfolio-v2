"""
Panel Optimizer Tool

태양광 패널 각도 최적화 Tool
"""

from typing import Dict, Any, Optional
from datetime import datetime
import math
from .base_tool import BaseTool


class PanelOptimizerTool(BaseTool):
    """태양광 패널 각도 최적화 Tool"""

    name = "optimize_panel_angle"
    description = "위치와 날짜를 기반으로 태양광 패널의 최적 각도를 계산합니다."

    def execute(
        self,
        latitude: float = 37.5,
        longitude: float = 127.0,
        date: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """패널 각도 최적화

        Args:
            latitude: 위도
            longitude: 경도
            date: 날짜 (선택)

        Returns:
            최적화 결과
        """
        # 날짜 파싱
        if date:
            try:
                target_date = datetime.fromisoformat(date)
            except ValueError:
                target_date = datetime.now()
        else:
            target_date = datetime.now()

        # 태양 고도각 기반 최적 각도 계산
        day_of_year = target_date.timetuple().tm_yday
        declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))

        # 정오 기준 최적 각도
        optimal_angle = latitude - declination

        # 계절별 조정
        month = target_date.month
        if month in [6, 7, 8]:  # 여름
            seasonal_adjustment = 5
        elif month in [12, 1, 2]:  # 겨울
            seasonal_adjustment = -5
        else:
            seasonal_adjustment = 0

        final_angle = optimal_angle + seasonal_adjustment

        # 범위 제한 (15 ~ 60도)
        final_angle = max(15, min(60, final_angle))

        return {
            "latitude": latitude,
            "longitude": longitude,
            "date": target_date.strftime("%Y-%m-%d"),
            "optimal_angle_degrees": round(final_angle, 1),
            "solar_declination": round(declination, 2),
            "seasonal_adjustment": seasonal_adjustment,
            "adjustment_range": {
                "min": round(final_angle - 5, 1),
                "max": round(final_angle + 5, 1),
            },
            "expected_efficiency_gain": "8-12%",
            "recommendation": f"패널 각도를 {round(final_angle, 1)}도로 설정하세요.",
        }
