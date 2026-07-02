"""Transport registry."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any, cast

from bedrock_bridge.exceptions import ConfigurationError

from .base import ProviderTransport
from .openai_chat_completions import OpenAIChatCompletionsTransport

TRANSPORT_ENTRY_POINT_GROUP = "bedrock_bridge.transports"


def list_transport_ids() -> list[str]:
    transport_ids = {"openai_chat_completions"}
    transport_ids.update(
        entry_point.name
        for entry_point in metadata.entry_points().select(group=TRANSPORT_ENTRY_POINT_GROUP)
    )
    return sorted(transport_ids)


def build_transport(
    transport_id: str,
    **kwargs: Any,
) -> ProviderTransport:
    if transport_id == "openai_chat_completions":
        return OpenAIChatCompletionsTransport(**kwargs)
    matches = [
        entry_point
        for entry_point in metadata.entry_points().select(group=TRANSPORT_ENTRY_POINT_GROUP)
        if entry_point.name == transport_id
    ]
    if len(matches) > 1:
        raise ConfigurationError(f"Ambiguous transport entry point: {transport_id}")
    if matches:
        transport_factory = cast(Callable[..., ProviderTransport], matches[0].load())
        return transport_factory(**kwargs)
    raise ConfigurationError(f"Unknown transport: {transport_id}")
