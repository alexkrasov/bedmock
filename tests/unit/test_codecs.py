from __future__ import annotations

import json

import pytest

from bedmock.canonical import CanonicalResponse, CanonicalTextBlock, CanonicalUsage
from bedmock.codecs import DEFAULT_CODEC_REGISTRY
from bedmock.exceptions import ValidationException


@pytest.mark.parametrize(
    ("model_id", "body", "codec_id"),
    [
        (
            "anthropic.claude-3-haiku-20240307-v1:0",
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "anthropic_messages",
        ),
        (
            "anthropic.claude-v2:1",
            {"prompt": "\n\nHuman: hi", "max_tokens_to_sample": 10},
            "anthropic_legacy",
        ),
        ("meta.llama3-8b-instruct-v1:0", {"prompt": "hi", "max_gen_len": 10}, "meta_llama"),
        ("mistral.mistral-7b-instruct-v0:2", {"prompt": "hi", "max_tokens": 10}, "mistral"),
        (
            "amazon.titan-text-express-v1",
            {"inputText": "hi", "textGenerationConfig": {"maxTokenCount": 10}},
            "amazon_titan_text",
        ),
        (
            "us.amazon.nova-2-lite-v1:0",
            {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            "amazon_nova",
        ),
        ("unknown.model", {"prompt": "hi", "max_tokens": 10}, "generic_prompt"),
    ],
)
def test_mandatory_codecs_decode_and_encode(
    model_id: str,
    body: dict[str, object],
    codec_id: str,
) -> None:
    codec = DEFAULT_CODEC_REGISTRY.detect(model_id, body)
    assert codec.codec_id == codec_id
    request = codec.decode_request(model_id, body)
    response = CanonicalResponse(
        id="resp",
        model="target",
        content=[CanonicalTextBlock("ok")],
        finish_reason="end_turn",
        usage=CanonicalUsage(1, 1, 2),
        provider_request_id="req",
    )
    encoded = codec.encode_response(request, response)
    assert isinstance(json.loads(json.dumps(encoded, default=str)), dict)


def test_anthropic_tool_and_image_validation() -> None:
    codec = DEFAULT_CODEC_REGISTRY.get("anthropic_messages")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "tool_choice": {"type": "tool", "name": "lookup"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "not-base64",
                        },
                    },
                ],
            }
        ],
    }
    with pytest.raises(ValidationException):
        codec.decode_request("anthropic.claude-3-haiku-20240307-v1:0", body)
