"""Mistral Bedrock text-generation codec."""

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


class MistralCodec:
    codec_id = "mistral"
    priority = 40

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return model_id.startswith("mistral.")

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValidationException("Mistral request requires string prompt")
        stop = body.get("stop") or body.get("stop_sequences") or []
        return CanonicalRequest(
            messages=[CanonicalMessage("user", [CanonicalTextBlock(prompt)])],
            system=[],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            stop_sequences=list(stop),
            tools=[],
            tool_choice=None,
            response_format=None,
            stream=False,
            metadata={"codec_id": self.codec_id},
            extensions={k: v for k, v in body.items() if k not in {"prompt", "max_tokens"}},
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        return {
            "outputs": [
                {"text": read_text(response.content), "stop_reason": response.finish_reason}
            ],
            "usage": usage_dict(response.usage),
        }

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"outputs": [{"text": event.text_delta}]}
        if event.event_type == "message_stop":
            return {"outputs": [{"text": "", "stop_reason": event.finish_reason}]}
        return None
