"""Converse operation codec."""

from __future__ import annotations

from typing import Any

from bedmock.canonical import CanonicalRequest, CanonicalResponse
from bedmock.codecs.utils import (
    converse_blocks_from_canonical,
    converse_content_to_blocks,
    messages_from_converse,
    response_format_from_extensions,
    tool_choice_from_converse,
    tools_from_converse,
)
from bedmock.exceptions import ValidationException

from .base import OperationContext, bedrock_response_metadata


class ConverseOperationCodec:
    operation_name = "Converse"

    def decode_operation_request(
        self,
        arguments: dict[str, Any],
        context: OperationContext,
    ) -> CanonicalRequest:
        model_id = arguments.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            raise ValidationException("converse requires modelId")
        if "messages" not in arguments:
            raise ValidationException("converse requires messages")
        if arguments.get("guardrailConfig"):
            raise ValidationException(
                "guardrailConfig is recognized but cannot be applied by bedmock"
            )
        inference = arguments.get("inferenceConfig") or {}
        if not isinstance(inference, dict):
            raise ValidationException("inferenceConfig must be an object")
        additional = arguments.get("additionalModelRequestFields") or {}
        if additional and not isinstance(additional, dict):
            raise ValidationException("additionalModelRequestFields must be an object")
        system = converse_content_to_blocks(arguments.get("system") or [])
        return CanonicalRequest(
            messages=messages_from_converse(arguments["messages"]),
            system=system,
            source_model_id=model_id,
            target_model=context.route.target_model,
            max_output_tokens=inference.get("maxTokens"),
            temperature=inference.get("temperature"),
            top_p=inference.get("topP"),
            top_k=inference.get("topK"),
            stop_sequences=list(inference.get("stopSequences") or []),
            tools=tools_from_converse(arguments.get("toolConfig")),
            tool_choice=tool_choice_from_converse(arguments.get("toolConfig")),
            response_format=response_format_from_extensions(additional),
            stream=False,
            metadata={
                "operation": self.operation_name,
                "codec_id": "converse",
                "route_id": context.route.route_id,
            },
            extensions={
                "additionalModelRequestFields": additional,
                "additionalModelResponseFieldPaths": arguments.get(
                    "additionalModelResponseFieldPaths"
                ),
                "requestMetadata": arguments.get("requestMetadata"),
                "performanceConfig": arguments.get("performanceConfig"),
                "serviceTier": arguments.get("serviceTier"),
            },
        )

    def encode_operation_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
        context: OperationContext,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": converse_blocks_from_canonical(response.content),
                }
            },
            "stopReason": response.finish_reason or "end_turn",
            "ResponseMetadata": bedrock_response_metadata(
                request_id=response.provider_request_id or context.request_id
            ),
        }
        usage: dict[str, int] = {}
        if response.usage.input_tokens is not None:
            usage["inputTokens"] = response.usage.input_tokens
        if response.usage.output_tokens is not None:
            usage["outputTokens"] = response.usage.output_tokens
        if response.usage.total_tokens is not None:
            usage["totalTokens"] = response.usage.total_tokens
        if response.usage.cached_input_tokens is not None:
            usage["cacheReadInputTokens"] = response.usage.cached_input_tokens
        if response.usage.cache_write_input_tokens is not None:
            usage["cacheWriteInputTokens"] = response.usage.cache_write_input_tokens
        if usage:
            payload["usage"] = usage
        if response.usage.reasoning_tokens is not None:
            payload.setdefault("metrics", {})["reasoningTokens"] = response.usage.reasoning_tokens
        return payload

    def encode_operation_stream(
        self,
        request: CanonicalRequest,
        events: Any,
        context: OperationContext,
    ) -> Any:
        raise NotImplementedError
