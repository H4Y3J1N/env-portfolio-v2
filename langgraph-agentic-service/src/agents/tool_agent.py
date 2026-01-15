"""
Tool Agent

외부 도구(ML 모델 등)를 호출하는 Agent
"""

import logging
import json
import re
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .state import AgentState

logger = logging.getLogger(__name__)


class ToolAgent:
    """Tool Calling Agent

    사용자 요청을 분석하여 적절한 Tool을 선택하고 실행
    """

    TOOL_SELECTION_PROMPT = """당신은 농업 AI 도구 선택 전문가입니다.
사용자 요청에 가장 적합한 도구를 선택하고 파라미터를 추출하세요.

## 사용 가능한 도구
1. **predict_yield**: 작물 수확량 예측
   - 파라미터: crop_type (작물 종류), area_sqm (면적, 제곱미터), planting_date (파종일, 선택)

2. **predict_temperature**: 기온 예측
   - 파라미터: datetime (예측 일시), location (위치, 선택)

3. **optimize_panel_angle**: 태양광 패널 각도 최적화
   - 파라미터: latitude (위도), longitude (경도), date (날짜, 선택)

4. **get_weather**: 현재/예보 날씨 조회
   - 파라미터: location (위치), date (날짜, 선택)

## 사용자 요청
{user_input}

## 응답 형식 (JSON)
도구 호출이 필요하면:
{{"tool_name": "도구이름", "arguments": {{"param1": "value1", ...}}}}

도구 호출이 필요없으면:
{{"tool_name": null, "arguments": {{}}}}

JSON만 출력하세요:"""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[Dict[str, Any]] = None,
        model_name: str = "gpt-4o-mini",
    ):
        """
        Args:
            llm: LLM 인스턴스
            tools: 사용 가능한 Tool 딕셔너리
            model_name: 사용할 모델 이름
        """
        self.llm = llm or ChatOpenAI(model=model_name, temperature=0)
        self.tools = tools or self._get_default_tools()
        self.prompt = ChatPromptTemplate.from_template(self.TOOL_SELECTION_PROMPT)

    def _get_default_tools(self) -> Dict[str, callable]:
        """기본 Tool 정의"""
        return {
            "predict_yield": self._predict_yield,
            "predict_temperature": self._predict_temperature,
            "optimize_panel_angle": self._optimize_panel_angle,
            "get_weather": self._get_weather,
        }

    def execute(self, state: AgentState) -> AgentState:
        """Tool 실행

        Args:
            state: 현재 Agent 상태

        Returns:
            Tool 실행 결과가 포함된 업데이트된 상태
        """
        user_input = state["user_input"]
        logger.info(f"Tool Agent executing for: {user_input[:50]}...")

        try:
            # LLM으로 Tool 선택
            chain = self.prompt | self.llm
            response = chain.invoke({"user_input": user_input})

            # JSON 파싱
            response_text = response.content if hasattr(response, 'content') else str(response)
            tool_call = self._parse_json_response(response_text)

            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments", {})

            if not tool_name:
                logger.info("No tool call needed")
                return state

            logger.info(f"Calling tool: {tool_name} with args: {arguments}")

            # Tool 실행
            if tool_name in self.tools:
                result = self.tools[tool_name](**arguments)
                success = True
                error = None
            else:
                result = None
                success = False
                error = f"Unknown tool: {tool_name}"

            # 결과 저장
            tool_result = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": success,
                "error": error,
            }

            existing_results = state.get("tool_results", [])

            return {
                **state,
                "tool_results": existing_results + [tool_result],
            }

        except Exception as e:
            logger.error(f"Tool Agent error: {e}")
            return {
                **state,
                "error": f"Tool 실행 오류: {str(e)}",
            }

    def _parse_json_response(self, response: str) -> dict:
        """LLM 응답에서 JSON 추출"""
        try:
            # JSON 블록 찾기
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"tool_name": None, "arguments": {}}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response}")
            return {"tool_name": None, "arguments": {}}

    # ============ Tool 구현 ============

    def _predict_yield(
        self,
        crop_type: str,
        area_sqm: float,
        planting_date: Optional[str] = None,
    ) -> dict:
        """수확량 예측 (Mock)"""
        # 실제로는 ML 모델 호출
        yield_per_sqm = {
            "토마토": 8.0,
            "tomato": 8.0,
            "고추": 3.5,
            "pepper": 3.5,
            "배추": 6.0,
            "cabbage": 6.0,
        }

        crop_lower = crop_type.lower()
        base_yield = yield_per_sqm.get(crop_lower, 5.0)
        total_yield = base_yield * area_sqm / 1000  # kg → ton

        return {
            "crop_type": crop_type,
            "area_sqm": area_sqm,
            "predicted_yield_ton": round(total_yield, 2),
            "yield_per_sqm_kg": base_yield,
            "confidence": 0.85,
        }

    def _predict_temperature(
        self,
        datetime: str,
        location: str = "서울",
    ) -> dict:
        """기온 예측 (Mock)"""
        # 실제로는 날씨 API 또는 ML 모델 호출
        return {
            "datetime": datetime,
            "location": location,
            "predicted_temp_celsius": 15.5,
            "min_temp": 10.0,
            "max_temp": 20.0,
            "confidence": 0.82,
        }

    def _optimize_panel_angle(
        self,
        latitude: float = 37.5,
        longitude: float = 127.0,
        date: Optional[str] = None,
    ) -> dict:
        """패널 각도 최적화 (Mock)"""
        # 간단한 계산 (실제로는 태양 위치 계산 필요)
        optimal_angle = 90 - latitude + 10  # 대략적인 계산

        return {
            "latitude": latitude,
            "longitude": longitude,
            "optimal_angle_degrees": round(optimal_angle, 1),
            "adjustment_range": {"min": optimal_angle - 5, "max": optimal_angle + 5},
            "expected_efficiency_gain": "12%",
        }

    def _get_weather(
        self,
        location: str = "서울",
        date: Optional[str] = None,
    ) -> dict:
        """날씨 조회 (Mock)"""
        return {
            "location": location,
            "date": date or "오늘",
            "condition": "맑음",
            "temperature": {"current": 18, "min": 12, "max": 22},
            "humidity": 65,
            "precipitation_probability": 10,
        }


class MockToolAgent(ToolAgent):
    """테스트용 Mock Tool Agent"""

    def execute(self, state: AgentState) -> AgentState:
        """항상 수확량 예측 결과 반환"""
        mock_result = {
            "tool_name": "predict_yield",
            "arguments": {"crop_type": "토마토", "area_sqm": 1000},
            "result": {
                "crop_type": "토마토",
                "area_sqm": 1000,
                "predicted_yield_ton": 8.0,
                "confidence": 0.85,
            },
            "success": True,
            "error": None,
        }

        existing_results = state.get("tool_results", [])

        return {
            **state,
            "tool_results": existing_results + [mock_result],
        }
