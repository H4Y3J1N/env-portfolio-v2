"""
Redis Session Store

Redis 기반 세션 저장소
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationSession(BaseModel):
    """대화 세션"""

    id: str = Field(..., description="세션 ID")
    tenant_id: str = Field("default", description="테넌트 ID")
    messages: List[Dict] = Field(default_factory=list, description="대화 히스토리")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)


class RedisSessionStore:
    """Redis 기반 세션 저장소

    Multi-turn 대화 히스토리 관리
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        session_ttl: int = 86400,  # 24시간
        key_prefix: str = "session:",
    ):
        """
        Args:
            redis_url: Redis 연결 URL
            session_ttl: 세션 TTL (초)
            key_prefix: 키 prefix
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.session_ttl = session_ttl
        self.key_prefix = key_prefix
        self.redis = None
        self._use_memory_fallback = False

        # 메모리 fallback 저장소
        self._memory_store: Dict[str, ConversationSession] = {}

    async def _get_redis(self):
        """Redis 클라이언트 가져오기"""
        if self._use_memory_fallback:
            return None

        if self.redis is None:
            try:
                import redis.asyncio as redis
                self.redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # 연결 테스트
                await self.redis.ping()
                logger.info("Redis connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory fallback.")
                self._use_memory_fallback = True
                return None

        return self.redis

    async def ping(self) -> bool:
        """Redis 연결 확인"""
        redis = await self._get_redis()
        if redis:
            await redis.ping()
            return True
        return False

    async def get(self, session_id: str) -> Optional[ConversationSession]:
        """세션 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 객체 또는 None
        """
        redis = await self._get_redis()

        if redis:
            key = f"{self.key_prefix}{session_id}"
            data = await redis.get(key)

            if data is None:
                return None

            return ConversationSession(**json.loads(data))
        else:
            # 메모리 fallback
            return self._memory_store.get(session_id)

    async def create(
        self,
        session_id: str,
        tenant_id: str = "default",
    ) -> ConversationSession:
        """세션 생성

        Args:
            session_id: 세션 ID
            tenant_id: 테넌트 ID

        Returns:
            생성된 세션
        """
        session = ConversationSession(
            id=session_id,
            tenant_id=tenant_id,
            messages=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        await self._save(session)
        logger.info(f"Session created: {session_id}")

        return session

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str = "default",
    ) -> ConversationSession:
        """세션 조회 또는 생성

        Args:
            session_id: 세션 ID
            tenant_id: 테넌트 ID

        Returns:
            세션 객체
        """
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, tenant_id)
        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> ConversationSession:
        """메시지 추가

        Args:
            session_id: 세션 ID
            role: 역할 (user/assistant/system)
            content: 메시지 내용
            metadata: 추가 메타데이터

        Returns:
            업데이트된 세션
        """
        session = await self.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        session.messages.append(message)
        session.updated_at = datetime.utcnow()

        await self._save(session)
        logger.debug(f"Message added to session {session_id}: {role}")

        return session

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """최근 메시지 조회

        Args:
            session_id: 세션 ID
            limit: 최대 메시지 수

        Returns:
            메시지 리스트
        """
        session = await self.get(session_id)
        if session is None:
            return []

        return session.messages[-limit:]

    async def _save(self, session: ConversationSession):
        """세션 저장

        Args:
            session: 저장할 세션
        """
        redis = await self._get_redis()

        if redis:
            key = f"{self.key_prefix}{session.id}"
            await redis.setex(
                key,
                self.session_ttl,
                session.model_dump_json(),
            )
        else:
            # 메모리 fallback
            self._memory_store[session.id] = session

    async def delete(self, session_id: str):
        """세션 삭제

        Args:
            session_id: 세션 ID
        """
        redis = await self._get_redis()

        if redis:
            key = f"{self.key_prefix}{session_id}"
            await redis.delete(key)
        else:
            self._memory_store.pop(session_id, None)

        logger.info(f"Session deleted: {session_id}")

    async def list_sessions(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        """세션 목록 조회

        Args:
            tenant_id: 테넌트 ID 필터 (선택)
            limit: 최대 개수

        Returns:
            세션 ID 리스트
        """
        redis = await self._get_redis()

        if redis:
            pattern = f"{self.key_prefix}*"
            keys = []
            async for key in redis.scan_iter(pattern, count=limit):
                session_id = key.replace(self.key_prefix, "")
                keys.append(session_id)
                if len(keys) >= limit:
                    break
            return keys
        else:
            return list(self._memory_store.keys())[:limit]

    async def close(self):
        """Redis 연결 종료"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
