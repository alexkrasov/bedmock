"""InvokeModelWithResponseStream operation codec."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bedrock_bridge.canonical import CanonicalRequest, CanonicalStreamEvent
from bedrock_bridge.codecs.registry import DEFAULT_CODEC_REGISTRY, CodecRegistry

from .base import BedrockEventStream, OperationContext, bedrock_response_metadata, json_event
from .invoke_model import InvokeModelOperationCodec


class InvokeModelWithResponseStreamOperationCodec(InvokeModelOperationCodec):
    operation_name = "InvokeModelWithResponseStream"

    def __init__(self, codec_registry: CodecRegistry | None = None) -> None:
        super().__init__(codec_registry or DEFAULT_CODEC_REGISTRY)

    def decode_operation_request(
        self,
        arguments: dict[str, Any],
        context: OperationContext,
    ) -> CanonicalRequest:
        request = super().decode_operation_request(arguments, context)
        request.stream = True
        request.metadata["operation"] = self.operation_name
        return request

    def encode_operation_stream(
        self,
        request: CanonicalRequest,
        events: Iterator[CanonicalStreamEvent],
        context: OperationContext,
    ) -> dict[str, Any]:
        codec = self.codec_registry.get(str(request.metadata["codec_id"]))

        def encode() -> Iterator[dict[str, Any]]:
            for event in events:
                payload = codec.encode_stream_event(request, event)
                if payload is not None:
                    yield json_event(payload)

        return {
            "body": BedrockEventStream(encode()),
            "contentType": "application/json",
            "ResponseMetadata": bedrock_response_metadata(request_id=context.request_id),
        }
