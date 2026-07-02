"""Canonical content blocks shared by operations, codecs, and transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CanonicalTextBlock:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalImageBlock:
    media_type: str
    data_base64: str | None = None
    url: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.data_base64) == bool(self.url):
            raise ValueError("Exactly one of data_base64 or url must be set for image blocks")


@dataclass
class CanonicalToolUseBlock:
    id: str
    name: str
    arguments: dict[str, Any] | str
    provider_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalToolResultBlock:
    tool_use_id: str
    content: list[CanonicalContentBlock]
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalJsonBlock:
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalReasoningBlock:
    text: str
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


CanonicalContentBlock = (
    CanonicalTextBlock
    | CanonicalImageBlock
    | CanonicalToolUseBlock
    | CanonicalToolResultBlock
    | CanonicalJsonBlock
    | CanonicalReasoningBlock
)

CanonicalRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class CanonicalMessage:
    role: CanonicalRole
    content: list[CanonicalContentBlock]
    metadata: dict[str, Any] = field(default_factory=dict)
