"""
Yield Predictor Tool

작물 수확량 예측 Tool
"""

from typing import Dict, Any, Optional
from .base_tool import BaseTool


class YieldPredictorTool(BaseTool):
    """작물 수확량 예측 Tool"""

    name = "predict_yield"
    description = "작물 종류와 면적을 기반으로 예상 수확량을 예측합니다."

    # 작물별 기준 수확량 (kg/m2)
    YIELD_PER_SQM = {
        "토마토": 8.0,
        "tomato": 8.0,
        "고추": 3.5,
        "pepper": 3.5,
        "배추": 6.0,
        "cabbage": 6.0,
        "감자": 4.0,
        "potato": 4.0,
        "양파": 5.5,
        "onion": 5.5,
        "마늘": 2.0,
        "garlic": 2.0,
    }

    def execute(
        self,
        crop_type: str,
        area_sqm: float,
        planting_date: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """수확량 예측

        Args:
            crop_type: 작물 종류
            area_sqm: 재배 면적 (제곱미터)
            planting_date: 파종일 (선택)

        Returns:
            예측 결과
        """
        crop_lower = crop_type.lower()
        base_yield = self.YIELD_PER_SQM.get(crop_lower, 5.0)

        # 면적 기반 총 수확량 계산
        total_yield_kg = base_yield * area_sqm
        total_yield_ton = total_yield_kg / 1000

        # 신뢰도 (실제로는 모델 기반)
        confidence = 0.85

        return {
            "crop_type": crop_type,
            "area_sqm": area_sqm,
            "planting_date": planting_date,
            "yield_per_sqm_kg": base_yield,
            "predicted_yield_kg": round(total_yield_kg, 2),
            "predicted_yield_ton": round(total_yield_ton, 2),
            "confidence": confidence,
            "unit": "ton",
        }
