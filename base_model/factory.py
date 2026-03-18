from __future__ import annotations

from typing import Any, Dict

from .mock_model import MockModel
from .openai_compatible import OpenAICompatibleModel


def build_model(config: Dict[str, Any]):
    model_cfg = config.get("model", {})
    provider = str(model_cfg.get("provider", "mock")).lower()

    if provider == "mock":
        return MockModel(config)

    if provider in {"openai", "openai_compatible", "deepseek", "openrouter"}:
        return OpenAICompatibleModel(config)

    raise ValueError(f"不支持的 model.provider: {provider}")