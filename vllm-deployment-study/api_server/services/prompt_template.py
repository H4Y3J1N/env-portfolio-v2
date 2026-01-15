"""
Prompt Template Service

프롬프트 템플릿 관리 및 포맷팅
"""

import json
from typing import Any, Dict, List, Optional


# ===========================================
# Available Tools Definition
# ===========================================
AVAILABLE_TOOLS: Dict[str, Dict[str, Any]] = {
    "predict_yield": {
        "description": "작물의 예상 수확량을 예측합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "crop_type": {
                    "type": "string",
                    "description": "작물 종류",
                    "enum": ["토마토", "고추", "상추", "배추", "오이"]
                },
                "area_m2": {
                    "type": "number",
                    "description": "재배 면적 (제곱미터)"
                },
                "planting_date": {
                    "type": "string",
                    "description": "파종일 (YYYY-MM-DD 형식, 선택사항)"
                }
            },
            "required": ["crop_type", "area_m2"]
        }
    },
    "predict_temperature": {
        "description": "특정 날짜의 기온을 예측합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "예측 날짜 (YYYY-MM-DD 형식)"
                },
                "location": {
                    "type": "string",
                    "description": "지역명"
                }
            },
            "required": ["date", "location"]
        }
    },
    "optimize_panel_angle": {
        "description": "태양광 패널의 최적 각도를 계산합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "위도 (예: 서울 37.5)"
                },
                "month": {
                    "type": "integer",
                    "description": "월 (1-12)"
                },
                "panel_type": {
                    "type": "string",
                    "description": "패널 유형",
                    "enum": ["fixed", "tracking"]
                }
            },
            "required": ["latitude", "month"]
        }
    },
    "get_weather": {
        "description": "특정 지역의 날씨 정보를 조회합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "지역명"
                },
                "date": {
                    "type": "string",
                    "description": "조회 날짜 (YYYY-MM-DD 형식, 생략 시 오늘)"
                }
            },
            "required": ["location"]
        }
    }
}


class PromptTemplate:
    """프롬프트 템플릿 관리 클래스"""

    # 기본 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 영농형 태양광 전문 AI 어시스턴트입니다.
농업과 태양광 발전에 대한 전문 지식을 바탕으로 사용자를 돕습니다.

당신은 다음과 같은 도구를 사용할 수 있습니다. 사용자의 요청에 따라 적절한 도구를 호출하세요.

도구를 호출할 때는 반드시 다음 형식을 사용하세요:
<tool_call>{"name": "도구이름", "arguments": {"인자1": "값1", "인자2": "값2"}}</tool_call>

정확하고 도움이 되는 답변을 제공하세요."""

    # Tool Calling 전용 시스템 프롬프트
    TOOL_CALLING_SYSTEM_PROMPT = """당신은 영농형 태양광 전문 AI 어시스턴트입니다.
사용자의 요청을 분석하여 적절한 도구를 호출해야 합니다.

## 도구 호출 규칙
1. 사용자 요청에 적합한 도구가 있으면 반드시 호출하세요.
2. 도구 호출 형식: <tool_call>{"name": "도구이름", "arguments": {"인자1": "값1"}}</tool_call>
3. arguments에는 도구의 필수 인자를 모두 포함해야 합니다.
4. 날짜는 YYYY-MM-DD 형식으로 작성하세요.

## 사용 가능한 도구"""

    def format_chat_prompt(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        일반 채팅 프롬프트 포맷팅 (Qwen2.5 형식)

        Args:
            user_message: 사용자 메시지
            history: 이전 대화 내역

        Returns:
            포맷팅된 프롬프트 문자열
        """
        # 도구 정보를 시스템 프롬프트에 추가
        tools_desc = self._format_tools_description(AVAILABLE_TOOLS)
        system_prompt = f"{self.SYSTEM_PROMPT}\n\n사용 가능한 도구:\n{tools_desc}"

        # Qwen2.5 채팅 형식으로 구성
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # 이전 대화 내역 추가
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # 현재 사용자 메시지 추가
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def format_tool_calling_prompt(
        self,
        user_message: str,
        available_tools: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Tool Calling 전용 프롬프트 포맷팅

        Args:
            user_message: 사용자 메시지
            available_tools: 사용 가능한 도구 목록

        Returns:
            포맷팅된 프롬프트 문자열
        """
        # 도구 상세 정보 포맷팅
        tools_json = self._format_tools_json(available_tools)

        system_prompt = f"{self.TOOL_CALLING_SYSTEM_PROMPT}\n{tools_json}"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def _format_tools_description(self, tools: Dict[str, Dict[str, Any]]) -> str:
        """도구 설명 포맷팅 (간략한 형태)"""
        lines = []
        for name, tool in tools.items():
            lines.append(f"- {name}: {tool['description']}")
        return "\n".join(lines)

    def _format_tools_json(self, tools: Dict[str, Dict[str, Any]]) -> str:
        """도구 JSON 포맷팅 (상세 형태)"""
        tools_list = []
        for name, tool in tools.items():
            tool_info = {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            tools_list.append(tool_info)

        return json.dumps(tools_list, ensure_ascii=False, indent=2)

    def format_tool_result_prompt(
        self,
        user_message: str,
        tool_name: str,
        tool_result: Any,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Tool 실행 결과를 포함한 프롬프트 포맷팅

        Args:
            user_message: 원본 사용자 메시지
            tool_name: 실행된 도구 이름
            tool_result: 도구 실행 결과
            history: 이전 대화 내역

        Returns:
            포맷팅된 프롬프트 문자열
        """
        system_prompt = """당신은 영농형 태양광 전문 AI 어시스턴트입니다.
도구 실행 결과를 바탕으로 사용자에게 친절하고 정확한 답변을 제공하세요."""

        tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # 이전 대화 내역
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # 사용자 메시지
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"

        # 도구 실행 결과
        prompt += f"<|im_start|>tool\n도구 '{tool_name}' 실행 결과:\n{tool_result_str}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"

        return prompt
