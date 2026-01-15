"""
Conversation Memory

LangChain 호환 대화 메모리 관리
"""

import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from .redis_store import RedisSessionStore

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Multi-turn 대화 메모리 관리

    Redis 세션 저장소와 연동하여 대화 히스토리 관리
    LangChain 메시지 형식과 호환
    """

    DEFAULT_SYSTEM_PROMPT = """당신은 농업 도메인 전문 AI 어시스턴트입니다.
사용자의 질문에 정확하고 도움이 되는 답변을 제공합니다.
필요한 경우 도구를 사용하거나 문서를 검색합니다.

다음 사항을 지켜주세요:
1. 검색된 문서 내용을 근거로 답변하세요
2. 확실하지 않은 내용은 솔직히 밝히세요
3. 한국어로 자연스럽게 답변하세요"""

    def __init__(
        self,
        session_store: Optional[RedisSessionStore] = None,
        max_history: int = 10,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            session_store: Redis 세션 저장소
            max_history: 유지할 최대 대화 턴 수
            system_prompt: 시스템 프롬프트
        """
        self.session_store = session_store or RedisSessionStore()
        self.max_history = max_history
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    async def get_messages_for_llm(
        self,
        session_id: str,
        current_message: Optional[str] = None,
        include_system: bool = True,
    ) -> List[BaseMessage]:
        """LLM 호출용 메시지 리스트 생성

        Args:
            session_id: 세션 ID
            current_message: 현재 사용자 메시지 (추가할 경우)
            include_system: 시스템 프롬프트 포함 여부

        Returns:
            LangChain BaseMessage 리스트
        """
        messages = []

        # 시스템 프롬프트
        if include_system:
            messages.append(SystemMessage(content=self.system_prompt))

        # 히스토리 로드
        history = await self.session_store.get_recent_messages(
            session_id,
            limit=self.max_history * 2,  # user + assistant 쌍
        )

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                # 시스템 메시지는 이미 추가했으므로 스킵
                pass

        # 현재 메시지 추가
        if current_message:
            messages.append(HumanMessage(content=current_message))

        return messages

    async def add_user_message(
        self,
        session_id: str,
        content: str,
    ):
        """사용자 메시지 추가

        Args:
            session_id: 세션 ID
            content: 메시지 내용
        """
        await self.session_store.get_or_create(session_id)
        await self.session_store.add_message(
            session_id,
            role="user",
            content=content,
        )

    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        """어시스턴트 메시지 추가

        Args:
            session_id: 세션 ID
            content: 메시지 내용
            metadata: 추가 메타데이터 (토큰 수 등)
        """
        await self.session_store.add_message(
            session_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )

    async def add_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[dict] = None,
    ):
        """대화 상호작용 추가 (user + assistant)

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            assistant_response: 어시스턴트 응답
            metadata: 추가 메타데이터
        """
        await self.add_user_message(session_id, user_message)
        await self.add_assistant_message(session_id, assistant_response, metadata)

    async def get_conversation_summary(
        self,
        session_id: str,
    ) -> str:
        """대화 요약 생성

        Args:
            session_id: 세션 ID

        Returns:
            대화 요약 문자열
        """
        history = await self.session_store.get_recent_messages(session_id)

        if not history:
            return "대화 히스토리가 없습니다."

        summary_parts = []
        for msg in history[-6:]:  # 최근 6개 메시지
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            summary_parts.append(f"[{role}] {content}...")

        return "\n".join(summary_parts)

    async def clear_history(self, session_id: str):
        """대화 히스토리 삭제

        Args:
            session_id: 세션 ID
        """
        await self.session_store.delete(session_id)
        logger.info(f"Conversation history cleared: {session_id}")

    def format_history_as_string(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """메시지 리스트를 문자열로 포맷팅

        Args:
            messages: BaseMessage 리스트

        Returns:
            포맷팅된 문자열
        """
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content[:100]}...")

        return "\n\n".join(parts)
