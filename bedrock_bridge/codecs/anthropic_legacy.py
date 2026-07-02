"""Anthropic legacy completion codec."""

from __future__ import annotations

from typing import Any

from bedrock_bridge.canonical import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalTextBlock,
)
from bedrock_bridge.exceptions import ValidationException

from .utils import read_text, usage_dict


class AnthropicLegacyCodec:
    codec_id = "anthropic_legacy"
    priority = 20

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return "prompt" in body and (
            "max_tokens_to_sample" in body or model_id.startswith("anthropic.")
        )

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValidationException("Anthropic legacy request requires string prompt")
        return CanonicalRequest(
            messages=[CanonicalMessage("user", [CanonicalTextBlock(prompt)])],
            system=[],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=body.get("max_tokens_to_sample"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            stop_sequences=list(body.get("stop_sequences") or []),
            tools=[],
            tool_choice=None,
            response_format=None,
            stream=False,
            metadata={"codec_id": self.codec_id},
            extensions={
                k: v for k, v in body.items() if k not in {"prompt", "max_tokens_to_sample"}
            },
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        return {
            "completion": read_text(response.content),
            "stop_reason": response.finish_reason,
            "stop": None,
            "model": request.source_model_id,
            "usage": usage_dict(response.usage),
        }

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"completion": event.text_delta, "stop_reason": None}
        if event.event_type == "message_stop":
            return {"completion": "", "stop_reason": event.finish_reason}
        return None
