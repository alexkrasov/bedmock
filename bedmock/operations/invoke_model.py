"""InvokeModel operation codec."""

from __future__ import annotations

from typing import Any

from bedmock.canonical import CanonicalRequest, CanonicalResponse
from bedmock.codecs.registry import DEFAULT_CODEC_REGISTRY, CodecRegistry
from bedmock.exceptions import ValidationException

from .base import (
    OperationContext,
    bedrock_response_metadata,
    read_body_as_json,
    streaming_body_from_json,
)


class InvokeModelOperationCodec:
    operation_name = "InvokeModel"

    def __init__(self, codec_registry: CodecRegistry | None = None) -> None:
        self.codec_registry = codec_registry or DEFAULT_CODEC_REGISTRY

    def decode_operation_request(
        self,
        arguments: dict[str, Any],
        context: OperationContext,
    ) -> CanonicalRequest:
        content_type = arguments.get("contentType", "application/json")
        if content_type != "application/json":
            raise ValidationException("Only application/json invoke_model contentType is supported")
        model_id = arguments.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            raise ValidationException("invoke_model requires modelId")
        if "body" not in arguments:
            raise ValidationException("invoke_model requires body")
        payload = read_body_as_json(arguments["body"])
        codec = self.codec_registry.detect(model_id, payload, context.route.source_codec)
        request = codec.decode_request(model_id, payload)
        request.target_model = context.route.target_model
        request.stream = False
        request.metadata.update(
            {
                "operation": self.operation_name,
                "codec_id": codec.codec_id,
                "accept": arguments.get("accept", "application/json"),
                "route_id": context.route.route_id,
            }
        )
        return request

    def encode_operation_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
        context: OperationContext,
    ) -> dict[str, Any]:
        codec = self.codec_registry.get(str(request.metadata["codec_id"]))
        payload = codec.encode_response(request, response)
        return {
            "body": streaming_body_from_json(payload),
            "contentType": "application/json",
            "ResponseMetadata": bedrock_response_metadata(
                request_id=response.provider_request_id or context.request_id
            ),
        }

    def encode_operation_stream(
        self,
        request: CanonicalRequest,
        events: Any,
        context: OperationContext,
    ) -> Any:
        raise NotImplementedError
