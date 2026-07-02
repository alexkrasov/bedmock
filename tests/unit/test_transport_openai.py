from __future__ import annotations

import json

import httpx
import pytest

from bedmock.canonical import (
    CanonicalImageBlock,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponseFormat,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from bedmock.canonical.usage import CanonicalUsage
from bedmock.exceptions import ServiceUnavailableException, UnsupportedOperationException
from bedmock.provider_profiles import load_provider_profile
from bedmock.transports.openai_chat_completions import OpenAIChatCompletionsTransport


def _request() -> CanonicalRequest:
    return CanonicalRequest(
        messages=[
            CanonicalMessage(
                "user",
                [
                    CanonicalTextBlock("describe"),
                    CanonicalImageBlock("image/png", data_base64="aGVsbG8="),
                ],
            )
        ],
        system=[CanonicalTextBlock("be brief")],
        source_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        target_model="gpt-test",
        max_output_tokens=12,
        temperature=0.0,
        top_p=0.9,
        top_k=5,
        stop_sequences=["END"],
        tools=[CanonicalTool("lookup", "Lookup", {"type": "object"})],
        tool_choice=CanonicalToolChoice("specific", tool_name="lookup"),
        response_format=CanonicalResponseFormat(
            "json_schema",
            schema={"type": "object"},
            name="answer",
            strict=True,
        ),
        stream=False,
        metadata={"operation": "InvokeModel"},
        extensions={},
    )


def test_openai_transport_payload_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "req-provider"},
            json={
                "id": "chatcmpl",
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 1},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        strict_parameters=False,
    )
    response = transport.invoke(_request(), load_provider_profile("openai"), "gpt-test")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["max_tokens"] == 12
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png")
    assert payload["tool_choice"]["function"]["name"] == "lookup"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert response.content[0].text == "done"
    assert response.usage == CanonicalUsage(10, 2, 12, reasoning_tokens=0, cached_input_tokens=1)


def test_openai_transport_count_tokens_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"input_tokens": 37})

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    count = transport.count_tokens(_request(), load_provider_profile("openai"), "gpt-test")

    payload = captured["json"]
    headers = captured["headers"]
    assert count == 37
    assert captured["url"] == "https://api.openai.com/v1/responses/input_tokens"
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer sk-test"
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-test"
    assert payload["instructions"] == "be brief"
    assert payload["input"][0]["role"] == "user"
    assert payload["input"][0]["content"][0] == {"type": "input_text", "text": "describe"}
    assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/png")
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup",
            "parameters": {"type": "object"},
        }
    ]
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True


def test_openai_transport_count_tokens_uses_gemini_native_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"totalTokens": 44})

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    count = transport.count_tokens(_request(), load_provider_profile("gemini"), "gemini-test")

    payload = captured["json"]
    headers = captured["headers"]
    assert count == 44
    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:countTokens"
    )
    assert isinstance(headers, dict)
    assert headers["x-goog-api-key"] == "gemini-test"
    assert isinstance(payload, dict)
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0] == {"text": "describe"}
    assert payload["contents"][0]["parts"][1]["inlineData"] == {
        "mimeType": "image/png",
        "data": "aGVsbG8=",
    }
    assert payload["systemInstruction"] == {
        "role": "system",
        "parts": [{"text": "be brief"}],
    }
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }
    assert payload["generationConfig"]["responseSchema"] == {"type": "object"}


def test_openai_transport_count_tokens_requires_exact_strategy() -> None:
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
    )

    with pytest.raises(UnsupportedOperationException) as exc_info:
        transport.count_tokens(_request(), load_provider_profile("openrouter"), "openai/gpt-4")

    assert "Exact token counting is not configured" in str(exc_info.value)


def test_openai_transport_preserves_provider_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"message": "Service Unavailable"}},
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )

    with pytest.raises(ServiceUnavailableException) as exc_info:
        transport.invoke(_request(), load_provider_profile("openai"), "gpt-test")

    error = exc_info.value.response["Error"]
    metadata = exc_info.value.response["ResponseMetadata"]
    assert error["Code"] == "ServiceUnavailableException"
    assert metadata["HTTPStatusCode"] == 503


def test_openai_transport_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                'data: {"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},'
                '"choices":[]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    events = list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))
    assert [event.event_type for event in events][:3] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]
    assert events[-1].event_type == "metadata"
    assert events[-1].usage.total_tokens == 2
