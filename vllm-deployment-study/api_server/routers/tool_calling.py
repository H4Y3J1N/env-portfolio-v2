"""
Tool Calling API Router

농업 도메인 Tool Calling API 엔드포인트
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.inference_client import InferenceClient
from services.prompt_template import PromptTemplate, AVAILABLE_TOOLS

logger = logging.getLogger(__name__)
router = APIRouter()


# ===========================================
# Request/Response Models
# ===========================================
class ToolCallRequest(BaseModel):
    """Tool Calling 요청"""
    query: str = Field(..., description="사용자 질의")
    available_tools: Optional[List[str]] = Field(
        default=None,
        description="사용 가능한 도구 목록 (None이면 전체)"
    )
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


class ToolCall(BaseModel):
    """Tool Call 결과"""
    name: str = Field(..., description="호출된 도구 이름")
    arguments: Dict[str, Any] = Field(..., description="도구 인자")


class ToolCallResponse(BaseModel):
    """Tool Calling 응답"""
    query: str = Field(..., description="원본 질의")
    tool_call: Optional[ToolCall] = Field(default=None, description="추출된 Tool Call")
    raw_response: str = Field(..., description="모델의 원본 응답")
    success: bool = Field(..., description="Tool Call 추출 성공 여부")
    error: Optional[str] = Field(default=None, description="에러 메시지")
    latency_ms: float = Field(..., description="처리 시간 (ms)")


class ToolInfo(BaseModel):
    """도구 정보"""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolExecuteRequest(BaseModel):
    """도구 실행 요청 (시뮬레이션)"""
    tool_name: str = Field(..., description="실행할 도구 이름")
    arguments: Dict[str, Any] = Field(..., description="도구 인자")


class ToolExecuteResponse(BaseModel):
    """도구 실행 응답"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    simulated: bool = True


# ===========================================
# Helper Functions
# ===========================================
def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """응답에서 Tool Call 추출"""
    # <tool_call>...</tool_call> 패턴 매칭
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None

    try:
        tool_call_str = match.group(1).strip()
        tool_call = json.loads(tool_call_str)

        # 필수 필드 검증
        if "name" not in tool_call or "arguments" not in tool_call:
            return None

        return tool_call

    except json.JSONDecodeError:
        return None


def simulate_tool_execution(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """도구 실행 시뮬레이션"""

    if tool_name == "predict_yield":
        crop_type = arguments.get("crop_type", "unknown")
        area_m2 = arguments.get("area_m2", 100)

        # 작물별 기본 수확량 (kg/100m2)
        yield_rates = {
            "토마토": 350,
            "고추": 175,
            "상추": 275,
            "배추": 300,
            "오이": 400
        }

        base_yield = yield_rates.get(crop_type, 200)
        estimated_yield = base_yield * (area_m2 / 100)

        return {
            "crop_type": crop_type,
            "area_m2": area_m2,
            "estimated_yield_kg": round(estimated_yield, 1),
            "confidence": 0.85
        }

    elif tool_name == "predict_temperature":
        date = arguments.get("date", "unknown")
        location = arguments.get("location", "서울")

        return {
            "location": location,
            "date": date,
            "predicted_high": 25,
            "predicted_low": 15,
            "confidence": 0.78
        }

    elif tool_name == "optimize_panel_angle":
        latitude = arguments.get("latitude", 37)
        month = arguments.get("month", 6)

        # 간단한 각도 계산 (실제로는 더 복잡)
        base_angle = latitude - 10
        seasonal_adjustment = 15 if month in [11, 12, 1, 2] else -10 if month in [5, 6, 7, 8] else 0
        optimal_angle = base_angle + seasonal_adjustment

        return {
            "latitude": latitude,
            "month": month,
            "optimal_angle": round(optimal_angle, 1),
            "direction": "정남향",
            "expected_efficiency_gain": "12%"
        }

    elif tool_name == "get_weather":
        location = arguments.get("location", "서울")
        date = arguments.get("date")

        return {
            "location": location,
            "date": date or "today",
            "condition": "맑음",
            "temperature": {"high": 26, "low": 18},
            "humidity": 55,
            "precipitation_probability": 10
        }

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ===========================================
# Endpoints
# ===========================================
@router.post("/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """
    Tool Calling API

    사용자 질의에서 적절한 도구 호출을 추출합니다.

    **예시 요청:**
    ```json
    {
        "query": "내일 서울 날씨를 알려줘",
        "available_tools": ["get_weather", "predict_temperature"]
    }
    ```

    **예시 응답:**
    ```json
    {
        "query": "내일 서울 날씨를 알려줘",
        "tool_call": {
            "name": "get_weather",
            "arguments": {"location": "서울", "date": "2025-01-15"}
        },
        "success": true
    }
    ```
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Tool call request: {request.query[:50]}...")

    # 사용 가능한 도구 필터링
    if request.available_tools:
        tools = {k: v for k, v in AVAILABLE_TOOLS.items() if k in request.available_tools}
    else:
        tools = AVAILABLE_TOOLS

    if not tools:
        raise HTTPException(status_code=400, detail="No valid tools specified")

    # 프롬프트 생성
    template = PromptTemplate()
    prompt = template.format_tool_calling_prompt(
        user_message=request.query,
        available_tools=tools
    )

    # Inference 서버 호출
    client = InferenceClient()

    try:
        result = await client.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>"]
        )

        raw_response = result["text"].strip()

        # Tool Call 추출
        tool_call_data = extract_tool_call(raw_response)

        if tool_call_data:
            logger.info(f"[{request_id}] Tool call extracted: {tool_call_data['name']}")

            return ToolCallResponse(
                query=request.query,
                tool_call=ToolCall(
                    name=tool_call_data["name"],
                    arguments=tool_call_data["arguments"]
                ),
                raw_response=raw_response,
                success=True,
                latency_ms=result["latency_ms"]
            )
        else:
            logger.warning(f"[{request_id}] No tool call found in response")

            return ToolCallResponse(
                query=request.query,
                tool_call=None,
                raw_response=raw_response,
                success=False,
                error="No tool call found in response",
                latency_ms=result["latency_ms"]
            )

    except Exception as e:
        logger.error(f"[{request_id}] Tool call error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """
    도구 실행 (시뮬레이션)

    지정된 도구를 시뮬레이션 실행합니다.
    실제 외부 API 호출 없이 예시 결과를 반환합니다.
    """
    logger.info(f"Executing tool: {request.tool_name}")

    if request.tool_name not in AVAILABLE_TOOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {request.tool_name}. Available: {list(AVAILABLE_TOOLS.keys())}"
        )

    result = simulate_tool_execution(request.tool_name, request.arguments)

    return ToolExecuteResponse(
        tool_name=request.tool_name,
        arguments=request.arguments,
        result=result,
        simulated=True
    )


@router.get("/available", response_model=List[ToolInfo])
async def list_available_tools():
    """
    사용 가능한 도구 목록

    현재 서비스에서 사용 가능한 모든 도구 정보를 반환합니다.
    """
    tools = []
    for name, tool in AVAILABLE_TOOLS.items():
        tools.append(ToolInfo(
            name=name,
            description=tool["description"],
            parameters=tool["parameters"]
        ))

    return tools


@router.get("/available/{tool_name}", response_model=ToolInfo)
async def get_tool_info(tool_name: str):
    """
    특정 도구 정보

    지정된 도구의 상세 정보를 반환합니다.
    """
    if tool_name not in AVAILABLE_TOOLS:
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}"
        )

    tool = AVAILABLE_TOOLS[tool_name]

    return ToolInfo(
        name=tool_name,
        description=tool["description"],
        parameters=tool["parameters"]
    )
