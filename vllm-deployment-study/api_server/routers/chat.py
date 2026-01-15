"""
Chat API Router

채팅 완성 API 엔드포인트
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.inference_client import InferenceClient
from services.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)
router = APIRouter()


# ===========================================
# Request/Response Models
# ===========================================
class Message(BaseModel):
    """대화 메시지"""
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청"""
    message: str = Field(..., description="사용자 메시지")
    conversation_id: Optional[str] = Field(default=None, description="대화 ID")
    history: Optional[List[Message]] = Field(default=None, description="이전 대화 내역")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="최대 생성 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="샘플링 온도")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 샘플링")


class ChatCompletionResponse(BaseModel):
    """채팅 완성 응답"""
    response: str = Field(..., description="생성된 응답")
    conversation_id: str = Field(..., description="대화 ID")
    tokens_used: int = Field(..., description="사용된 토큰 수")
    latency_ms: float = Field(..., description="처리 시간 (ms)")


class SimpleMessage(BaseModel):
    """간단한 메시지 요청"""
    message: str = Field(..., description="사용자 메시지")


# ===========================================
# Endpoints
# ===========================================
@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    채팅 완성 API

    사용자 메시지에 대한 AI 응답을 생성합니다.

    **예시 요청:**
    ```json
    {
        "message": "토마토 100평 재배 시 예상 수확량을 알려주세요.",
        "max_tokens": 256,
        "temperature": 0.7
    }
    ```
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]

    logger.info(f"[{conversation_id}] Chat request: {request.message[:50]}...")

    # 프롬프트 생성
    template = PromptTemplate()
    prompt = template.format_chat_prompt(
        user_message=request.message,
        history=request.history
    )

    # Inference 서버 호출
    client = InferenceClient()

    try:
        result = await client.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["<|im_end|>"]
        )

        # 응답 텍스트 정리
        response_text = result["text"].strip()

        logger.info(f"[{conversation_id}] Response generated: {len(response_text)} chars")

        return ChatCompletionResponse(
            response=response_text,
            conversation_id=conversation_id,
            tokens_used=result["tokens_used"],
            latency_ms=result["latency_ms"]
        )

    except Exception as e:
        logger.error(f"[{conversation_id}] Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simple", response_model=ChatCompletionResponse)
async def simple_chat(request: SimpleMessage):
    """
    간단한 채팅 API

    기본 설정으로 빠르게 응답을 생성합니다.
    """
    full_request = ChatCompletionRequest(
        message=request.message,
        max_tokens=256,
        temperature=0.7
    )
    return await chat_completion(full_request)


@router.get("/models")
async def list_models():
    """
    사용 가능한 모델 목록

    현재 서비스에서 사용 가능한 모델 정보를 반환합니다.
    """
    client = InferenceClient()

    try:
        health = await client.health_check()
        model_name = health.get("model", "unknown")
        quantization = health.get("quantization", "unknown")

        return {
            "models": [
                {
                    "id": model_name,
                    "object": "model",
                    "quantization": quantization,
                    "owner": "local",
                    "capabilities": ["chat", "tool_calling"]
                }
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=503, detail="Inference server unavailable")
