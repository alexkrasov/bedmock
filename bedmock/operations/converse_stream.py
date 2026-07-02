"""ConverseStream operation codec."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bedmock.canonical import CanonicalRequest, CanonicalStreamEvent

from .base import BedrockEventStream, OperationContext, bedrock_response_metadata
from .converse import ConverseOperationCodec


class ConverseStreamOperationCodec(ConverseOperationCodec):
    operation_name = "ConverseStream"

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
        def encode() -> Iterator[dict[str, Any]]:
            for event in events:
                payload = self._encode_event(event)
                if payload is not None:
                    yield payload

        return {
            "stream": BedrockEventStream(encode()),
            "ResponseMetadata": bedrock_response_metadata(request_id=context.request_id),
        }

    def _encode_event(self, event: CanonicalStreamEvent) -> dict[str, Any] | None:
        index = event.content_block_index or 0
        if event.event_type == "message_start":
            return {"messageStart": {"role": "assistant"}}
        if event.event_type == "content_block_start":
            content_block_start: dict[str, Any] = {"contentBlockIndex": index, "start": {}}
            if event.content_block_type == "tool_use":
                content_block_start = {
                    "contentBlockIndex": index,
                    "start": {
                        "toolUse": {
                            "toolUseId": event.tool_call_id,
                            "name": event.tool_name,
                        }
                    },
                }
            return {"contentBlockStart": content_block_start}
        if event.event_type == "content_block_delta":
            delta: dict[str, Any]
            if event.reasoning_delta is not None:
                delta = {"reasoningContent": {"text": event.reasoning_delta}}
            elif event.tool_arguments_delta is not None:
                delta = {"toolUse": {"input": event.tool_arguments_delta}}
            else:
                delta = {"text": event.text_delta or ""}
            return {"contentBlockDelta": {"contentBlockIndex": index, "delta": delta}}
        if event.event_type == "content_block_stop":
            return {"contentBlockStop": {"contentBlockIndex": index}}
        if event.event_type == "message_stop":
            return {"messageStop": {"stopReason": event.finish_reason or "end_turn"}}
        if event.event_type == "metadata":
            usage: dict[str, int] = {}
            if event.usage:
                if event.usage.input_tokens is not None:
                    usage["inputTokens"] = event.usage.input_tokens
                if event.usage.output_tokens is not None:
                    usage["outputTokens"] = event.usage.output_tokens
                if event.usage.total_tokens is not None:
                    usage["totalTokens"] = event.usage.total_tokens
            return {"metadata": {"usage": usage, "metrics": event.metadata.get("metrics", {})}}
        if event.event_type == "error":
            return {"internalServerException": event.metadata}
        return None
