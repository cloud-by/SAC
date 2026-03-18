from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def generate(
            self,
            prompt: str,
            temperature: float = 0.2,
            max_tokens: int = 512,
            **kwargs: Any,
    ) -> str:
        """
        统一单轮文本生成接口，默认转发到 chat。
        """
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

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