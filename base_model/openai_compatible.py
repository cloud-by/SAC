from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from .base import BaseModel


class OpenAICompatibleModel(BaseModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        model_cfg = config.get("model", {})
        self.model_name = model_cfg.get("name", "gpt-4o-mini")
        self.base_url = model_cfg.get("base_url", "https://api.openai.com/v1")
        self.api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(self.api_key_env, "")

        if not self.api_key:
            raise ValueError(
                f"未找到环境变量 {self.api_key_env}，无法初始化 OpenAICompatibleModel"
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]