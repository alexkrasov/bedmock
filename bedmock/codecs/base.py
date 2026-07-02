"""Model-family codec protocol."""

from __future__ import annotations

from typing import Any, Protocol

from bedmock.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent


class BedrockModelCodec(Protocol):
    codec_id: str
    priority: int

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool: ...

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest: ...

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]: ...

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None: ...
