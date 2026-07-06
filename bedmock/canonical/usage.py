"""Canonical token usage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanonicalUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None

    @classmethod
    def empty(cls) -> CanonicalUsage:
        return cls(input_tokens=None, output_tokens=None, total_tokens=None)
