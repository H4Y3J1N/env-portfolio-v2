"""
Base Tool

Tool 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel


class BaseTool(ABC):
    """Tool 베이스 클래스"""

    name: str = "base_tool"
    description: str = "Base tool"

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Tool 실행"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Tool 스키마 반환"""
        return {
            "name": self.name,
            "description": self.description,
        }
