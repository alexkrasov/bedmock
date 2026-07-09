"""Canonical tool definitions and tool choice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CanonicalTool:
    name: str
    description: str | None
    input_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    strict: bool | None = None


@dataclass
class CanonicalToolChoice:
    mode: Literal["auto", "none", "required", "any", "specific"]
    tool_name: str | None = None
    disable_parallel_tool_calls: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode == "specific" and not self.tool_name:
            raise ValueError("tool_name is required for specific tool choice")
        if self.mode != "specific" and self.tool_name and not self.metadata.get("extension"):
            raise ValueError("tool_name is only valid for specific tool choice")


@dataclass
class CanonicalResponseFormat:
    mode: Literal["text", "json_object", "json_schema"]
    schema: dict[str, Any] | None = None
    name: str | None = None
    strict: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
