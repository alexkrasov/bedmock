"""Transport registry."""

from __future__ import annotations

from typing import Any

from bedrock_bridge.exceptions import ConfigurationError

from .base import ProviderTransport
from .openai_chat_completions import OpenAIChatCompletionsTransport


def list_transport_ids() -> list[str]:
    return ["openai_chat_completions"]


def build_transport(
    transport_id: str,
    **kwargs: Any,
) -> ProviderTransport:
    if transport_id == "openai_chat_completions":
        return OpenAIChatCompletionsTransport(**kwargs)
    raise ConfigurationError(f"Unknown transport: {transport_id}")
