"""Canonical response model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .content import CanonicalContentBlock
from .usage import CanonicalUsage


@dataclass
class CanonicalResponse:
    id: str
    model: str
    content: list[CanonicalContentBlock]
    finish_reason: str | None
    usage: CanonicalUsage
    provider_request_id: str | None
    raw_provider_response: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
