"""Amazon Titan Text codec."""

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


class AmazonTitanTextCodec:
    codec_id = "amazon_titan_text"
    priority = 50

    def can_decode(self, model_id: str, body: dict[str, Any]) -> bool:
        return model_id.startswith("amazon.titan") or "inputText" in body

    def decode_request(self, model_id: str, body: dict[str, Any]) -> CanonicalRequest:
        input_text = body.get("inputText")
        if not isinstance(input_text, str):
            raise ValidationException("Titan Text request requires string inputText")
        config = body.get("textGenerationConfig") or {}
        if not isinstance(config, dict):
            raise ValidationException("textGenerationConfig must be an object")
        return CanonicalRequest(
            messages=[CanonicalMessage("user", [CanonicalTextBlock(input_text)])],
            system=[],
            source_model_id=model_id,
            target_model=None,
            max_output_tokens=config.get("maxTokenCount"),
            temperature=config.get("temperature"),
            top_p=config.get("topP"),
            top_k=None,
            stop_sequences=list(config.get("stopSequences") or []),
            tools=[],
            tool_choice=None,
            response_format=None,
            stream=False,
            metadata={"codec_id": self.codec_id},
            extensions={
                k: v for k, v in body.items() if k not in {"inputText", "textGenerationConfig"}
            },
        )

    def encode_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "results": [
                {
                    "outputText": read_text(response.content),
                    "completionReason": response.finish_reason,
                }
            ]
        }
        if response.usage.input_tokens is not None:
            payload["inputTextTokenCount"] = response.usage.input_tokens
        if response.usage.output_tokens is not None:
            payload["results"][0]["tokenCount"] = response.usage.output_tokens
        return payload

    def encode_stream_event(
        self,
        request: CanonicalRequest,
        event: CanonicalStreamEvent,
    ) -> dict[str, Any] | None:
        if event.event_type == "content_block_delta" and event.text_delta is not None:
            return {"outputText": event.text_delta}
        if event.event_type == "message_stop":
            return {"completionReason": event.finish_reason}
        return None
