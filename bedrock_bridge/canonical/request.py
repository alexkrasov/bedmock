"""Canonical request model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .content import CanonicalContentBlock, CanonicalMessage
from .tools import CanonicalResponseFormat, CanonicalTool, CanonicalToolChoice


@dataclass
class CanonicalRequest:
    messages: list[CanonicalMessage]
    system: list[CanonicalContentBlock]
    source_model_id: str
    target_model: str | None
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    stop_sequences: list[str]
    tools: list[CanonicalTool]
    tool_choice: CanonicalToolChoice | None
    response_format: CanonicalResponseFormat | None
    stream: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
