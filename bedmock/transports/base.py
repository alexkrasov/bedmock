"""Provider transport protocol."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from bedmock.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from bedmock.provider_profiles import ProviderProfile


class ProviderTransport(Protocol):
    transport_id: str

    def invoke(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> CanonicalResponse: ...

    def invoke_stream(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> Iterator[CanonicalStreamEvent]: ...

    def count_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int: ...

    def close(self) -> None: ...
