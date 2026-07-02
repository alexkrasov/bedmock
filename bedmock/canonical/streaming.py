"""Canonical stream event model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .usage import CanonicalUsage


@dataclass
class CanonicalStreamEvent:
    event_type: Literal[
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "metadata",
        "error",
    ]
    sequence_number: int
    content_block_index: int | None = None
    content_block_type: str | None = None
    text_delta: str | None = None
    reasoning_delta: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments_delta: str | None = None
    finish_reason: str | None = None
    usage: CanonicalUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
