"""Amazon Nova messages codec."""

from __future__ import annotations

from typing import Any

from bedrock_bridge.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from bedrock_bridge.exceptions import ValidationException

from .utils import (
    converse_blocks_from_canonical,
    messages_from_converse,
    response_format_from_extensions,
    tool_choice_from_converse,
    tools_from_converse,
)


class AmazonNovaCodec:
    codec_id = "amazon_nova"
    priority = 15

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return model_id.startswith("amazon.nova") or model_id.startswith("us.amazon.nova")

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        if "messages" not in body:
            raise ValidationException("Nova request requires messages")
        inference = body.get("inferenceConfig") or {}
        if not isinstance(inference, dict):
            raise ValidationException("inferenceConfig must be object")
        return CanonicalRequest(
            messages=messages_from_converse(body["messages"]),
            system=messages_from_converse([{"role": "user", "content": body.get("system") or []}])[
                0
            ].content
            if body.get("system")
            else [],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=inference.get("maxTokens"),
            temperature=inference.get("temperature"),
            top_p=inference.get("topP"),
            top_k=inference.get("topK"),
            stop_sequences=list(inference.get("stopSequences") or []),
            tools=tools_from_converse(body.get("toolConfig")),
            tool_choice=tool_choice_from_converse(body.get("toolConfig")),
            response_format=response_format_from_extensions(body),
            stream=False,
            metadata={"codec_id": self.codec_id},
            extensions={
                k: v
                for k, v in body.items()
                if k not in {"messages", "system", "inferenceConfig", "toolConfig"}
            },
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": converse_blocks_from_canonical(response.content),
                }
            },
            "stopReason": response.finish_reason or "end_turn",
        }
        usage: dict[str, int] = {}
        if response.usage.input_tokens is not None:
            usage["inputTokens"] = response.usage.input_tokens
        if response.usage.output_tokens is not None:
            usage["outputTokens"] = response.usage.output_tokens
        if response.usage.total_tokens is not None:
            usage["totalTokens"] = response.usage.total_tokens
        if usage:
            payload["usage"] = usage
        return payload

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"contentBlockDelta": {"delta": {"text": event.text_delta}}}
        if event.event_type == "message_stop":
            return {"messageStop": {"stopReason": event.finish_reason or "end_turn"}}
        return None
