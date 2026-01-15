"""
Langfuse Tracker

토큰 사용량 및 LLM 트레이싱
"""

import logging
import os
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Langfuse 토큰 사용량 및 트레이싱

    LLM 호출, RAG 검색, Tool 실행 등을 추적
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            enabled: 추적 활성화 여부
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.enabled = enabled and bool(self.public_key and self.secret_key)

        self.langfuse = None
        self._init_langfuse()

    def _init_langfuse(self):
        """Langfuse 클라이언트 초기화"""
        if not self.enabled:
            logger.info("Langfuse tracking disabled")
            return

        try:
            from langfuse import Langfuse

            self.langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info("Langfuse initialized successfully")

        except ImportError:
            logger.warning("langfuse package not installed")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False

    def create_trace(
        self,
        name: str = "chat_completion",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """새 트레이스 생성

        Args:
            name: 트레이스 이름
            session_id: 세션 ID
            user_id: 사용자 ID
            metadata: 추가 메타데이터

        Returns:
            Langfuse trace 객체 또는 MockTrace
        """
        if not self.enabled or not self.langfuse:
            return MockTrace(name, session_id)

        try:
            trace = self.langfuse.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
            )
            logger.debug(f"Created trace: {name}")
            return trace

        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return MockTrace(name, session_id)

    def track_generation(
        self,
        trace,
        name: str,
        model: str,
        input_messages: List[Dict],
        output: str,
        usage: Dict[str, int],
        metadata: Optional[Dict] = None,
    ):
        """LLM 생성 추적

        Args:
            trace: 트레이스 객체
            name: 생성 이름
            model: 모델 이름
            input_messages: 입력 메시지
            output: 출력 텍스트
            usage: 토큰 사용량 {"prompt_tokens": N, "completion_tokens": M}
            metadata: 추가 메타데이터
        """
        if isinstance(trace, MockTrace):
            trace.add_generation(name, model, usage)
            return

        try:
            trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                metadata=metadata or {},
            )
            logger.debug(f"Tracked generation: {name}, tokens: {usage}")

        except Exception as e:
            logger.error(f"Failed to track generation: {e}")

    def track_retrieval(
        self,
        trace,
        name: str,
        query: str,
        documents: List[Dict],
        metadata: Optional[Dict] = None,
    ):
        """RAG 검색 추적

        Args:
            trace: 트레이스 객체
            name: span 이름
            query: 검색 쿼리
            documents: 검색된 문서
            metadata: 추가 메타데이터
        """
        if isinstance(trace, MockTrace):
            trace.add_span(name, "retrieval", {"num_docs": len(documents)})
            return

        try:
            trace.span(
                name=name,
                input={"query": query},
                output={
                    "num_documents": len(documents),
                    "top_scores": [d.get("score", 0) for d in documents[:3]],
                },
                metadata=metadata or {},
            )
            logger.debug(f"Tracked retrieval: {name}, docs: {len(documents)}")

        except Exception as e:
            logger.error(f"Failed to track retrieval: {e}")

    def track_tool_call(
        self,
        trace,
        tool_name: str,
        arguments: Dict,
        result: Any,
        success: bool = True,
        duration_ms: Optional[float] = None,
    ):
        """Tool 호출 추적

        Args:
            trace: 트레이스 객체
            tool_name: Tool 이름
            arguments: Tool 인자
            result: 실행 결과
            success: 성공 여부
            duration_ms: 실행 시간 (ms)
        """
        if isinstance(trace, MockTrace):
            trace.add_span(f"tool_{tool_name}", "tool", {"success": success})
            return

        try:
            trace.span(
                name=f"tool_{tool_name}",
                input=arguments,
                output=result,
                metadata={
                    "success": success,
                    "duration_ms": duration_ms,
                },
            )
            logger.debug(f"Tracked tool call: {tool_name}, success: {success}")

        except Exception as e:
            logger.error(f"Failed to track tool call: {e}")

    def track_agent_decision(
        self,
        trace,
        agent: str,
        decision: str,
        reasoning: str,
    ):
        """Agent 결정 추적

        Args:
            trace: 트레이스 객체
            agent: Agent 이름
            decision: 결정 내용
            reasoning: 결정 이유
        """
        if isinstance(trace, MockTrace):
            trace.add_span(f"agent_{agent}", "decision", {"decision": decision})
            return

        try:
            trace.span(
                name=f"agent_{agent}",
                input={"agent": agent},
                output={"decision": decision, "reasoning": reasoning},
            )

        except Exception as e:
            logger.error(f"Failed to track agent decision: {e}")

    def flush(self):
        """버퍼 플러시 (동기)"""
        if self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")

    async def aflush(self):
        """버퍼 플러시 (비동기)"""
        self.flush()

    def shutdown(self):
        """종료"""
        self.flush()
        if self.langfuse:
            try:
                self.langfuse.shutdown()
            except Exception:
                pass


class MockTrace:
    """Mock Trace (Langfuse 비활성화 시)"""

    def __init__(self, name: str, session_id: Optional[str] = None):
        self.name = name
        self.session_id = session_id
        self.spans = []
        self.generations = []

    def add_span(self, name: str, type: str, data: Dict):
        """Mock span 추가"""
        self.spans.append({
            "name": name,
            "type": type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def add_generation(self, name: str, model: str, usage: Dict):
        """Mock generation 추가"""
        self.generations.append({
            "name": name,
            "model": model,
            "usage": usage,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def generation(self, **kwargs):
        """Langfuse generation 호환 메서드"""
        self.add_generation(
            kwargs.get("name", "unknown"),
            kwargs.get("model", "unknown"),
            kwargs.get("usage", {}),
        )

    def span(self, **kwargs):
        """Langfuse span 호환 메서드"""
        self.add_span(
            kwargs.get("name", "unknown"),
            "span",
            kwargs.get("output", {}),
        )

    def get_summary(self) -> Dict:
        """추적 요약"""
        total_tokens = sum(
            g.get("usage", {}).get("total", 0)
            for g in self.generations
        )
        return {
            "name": self.name,
            "session_id": self.session_id,
            "num_spans": len(self.spans),
            "num_generations": len(self.generations),
            "total_tokens": total_tokens,
        }
