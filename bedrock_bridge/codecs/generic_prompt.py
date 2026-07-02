"""Restricted generic prompt fallback codec."""

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

from .utils import read_text


class GenericPromptCodec:
    codec_id = "generic_prompt"
    priority = 1000

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return any(key in body for key in ("prompt", "input", "text"))

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        prompt = body.get("prompt", body.get("input", body.get("text")))
        if not isinstance(prompt, str):
            raise ValidationException("Generic prompt codec requires prompt/input/text string")
        return CanonicalRequest(
            messages=[CanonicalMessage("user", [CanonicalTextBlock(prompt)])],
            system=[],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=body.get("max_tokens") or body.get("maxTokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p") or body.get("topP"),
            top_k=body.get("top_k") or body.get("topK"),
            stop_sequences=list(body.get("stop") or body.get("stop_sequences") or []),
            tools=[],
            tool_choice=None,
            response_format=None,
            stream=False,
            metadata={"codec_id": self.codec_id, "fallback": True},
            extensions={k: v for k, v in body.items() if k not in {"prompt", "input", "text"}},
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        return {"text": read_text(response.content), "finish_reason": response.finish_reason}

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"text": event.text_delta}
        return None
