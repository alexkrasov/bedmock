"""Anthropic Messages on Bedrock codec."""

from __future__ import annotations

from typing import Any

from bedmock.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamEvent,
)
from bedmock.canonical.usage import CanonicalUsage
from bedmock.exceptions import ValidationException

from .utils import (
    anthropic_blocks_from_canonical,
    anthropic_blocks_from_system,
    finish_reason_to_openai,
    messages_from_anthropic,
    response_format_from_extensions,
    tool_choice_from_anthropic,
    tools_from_anthropic,
    usage_dict,
)


class AnthropicMessagesCodec:
    codec_id = "anthropic_messages"
    priority = 10

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return "messages" in body and (
            "anthropic_version" in body or model_id.startswith("anthropic.")
        )

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        if "messages" not in body:
            raise ValidationException("Anthropic Messages request requires messages")
        return CanonicalRequest(
            messages=messages_from_anthropic(body["messages"]),
            system=anthropic_blocks_from_system(body.get("system")),
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            stop_sequences=list(body.get("stop_sequences") or []),
            tools=tools_from_anthropic(body.get("tools")),
            tool_choice=tool_choice_from_anthropic(body.get("tool_choice")),
            response_format=response_format_from_extensions(body),
            stream=False,
            metadata={
                "codec_id": self.codec_id,
                "anthropic_version": body.get("anthropic_version"),
            },
            extensions={
                key: value
                for key, value in body.items()
                if key
                not in {
                    "anthropic_version",
                    "messages",
                    "system",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                    "tools",
                    "tool_choice",
                    "response_format",
                }
            },
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        usage = usage_dict(response.usage)
        return {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "model": request.source_model_id,
            "content": anthropic_blocks_from_canonical(response.content),
            "stop_reason": response.finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            },
        }

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "message_start":
            return {
                "type": "message_start",
                "message": {
                    "id": event.metadata.get("id", "msg_bedmock_stream"),
                    "type": "message",
                    "role": "assistant",
                    "model": request.source_model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        if event.event_type == "content_block_start":
            block: dict[str, Any]
            if event.content_block_type == "tool_use":
                block = {
                    "type": "tool_use",
                    "id": event.tool_call_id,
                    "name": event.tool_name,
                    "input": {},
                }
            else:
                block = {"type": "text", "text": ""}
            return {
                "type": "content_block_start",
                "index": event.content_block_index or 0,
                "content_block": block,
            }
        if event.event_type == "content_block_delta":
            if event.tool_arguments_delta is not None:
                delta = {"type": "input_json_delta", "partial_json": event.tool_arguments_delta}
            else:
                delta = {"type": "text_delta", "text": event.text_delta or ""}
            return {
                "type": "content_block_delta",
                "index": event.content_block_index or 0,
                "delta": delta,
            }
        if event.event_type == "content_block_stop":
            return {"type": "content_block_stop", "index": event.content_block_index or 0}
        if event.event_type == "message_delta":
            usage = event.usage or CanonicalUsage.empty()
            return {
                "type": "message_delta",
                "delta": {"stop_reason": event.finish_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage.output_tokens},
            }
        if event.event_type == "message_stop":
            return {"type": "message_stop"}
        if event.event_type == "error":
            return {"type": "error", "error": event.metadata}
        return None


def canonical_finish_to_anthropic(reason: str | None) -> str | None:
    if reason == "tool_use":
        return "tool_use"
    return finish_reason_to_openai(reason)
