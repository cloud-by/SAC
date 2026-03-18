from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseModel


class MockModel(BaseModel):
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        return "MOCK_RESPONSE: 当前为 mock 模型返回，用于占位调试。"