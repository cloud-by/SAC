from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        统一聊天接口：
        messages 形如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        raise NotImplementedError