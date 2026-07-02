"""Meta Llama Bedrock text-generation codec."""

from __future__ import annotations

from typing import Any

from bedmock.canonical import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalTextBlock,
)
from bedmock.exceptions import ValidationException

from .utils import read_text


class MetaLlamaCodec:
    codec_id = "meta_llama"
    priority = 30

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return model_id.startswith("meta.") or "max_gen_len" in body

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValidationException("Meta Llama request requires string prompt")
        return CanonicalRequest(
            messages=[CanonicalMessage("user", [CanonicalTextBlock(prompt)])],
            system=[],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=body.get("max_gen_len"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=None,
            stop_sequences=list(body.get("stop") or []),
            tools=[],
            tool_choice=None,
            response_format=None,
            stream=False,
            metadata={"codec_id": self.codec_id},
            extensions={k: v for k, v in body.items() if k not in {"prompt", "max_gen_len"}},
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generation": read_text(response.content),
            "stop_reason": response.finish_reason,
        }
        if response.usage.input_tokens is not None:
            payload["prompt_token_count"] = response.usage.input_tokens
        if response.usage.output_tokens is not None:
            payload["generation_token_count"] = response.usage.output_tokens
        return payload

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"generation": event.text_delta}
        if event.event_type == "message_stop":
            return {"stop_reason": event.finish_reason}
        return None
