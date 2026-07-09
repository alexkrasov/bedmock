from __future__ import annotations

import json

import httpx
import pytest

from bedmock.canonical import (
    CanonicalCachePointBlock,
    CanonicalImageBlock,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponseFormat,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
)
from bedmock.canonical.usage import CanonicalUsage
from bedmock.exceptions import (
    AccessDeniedException,
    InternalServerException,
    ModelStreamErrorException,
    ServiceUnavailableException,
    UnsupportedOperationException,
    ValidationException,
)
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


def _cache_point_request(ttl: str | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        messages=[CanonicalMessage("user", [CanonicalTextBlock("What changed?")])],
        system=[
            CanonicalTextBlock("Shared policy document"),
            CanonicalCachePointBlock(ttl=ttl),
        ],
        source_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        target_model="target-test",
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        stop_sequences=[],
        tools=[],
        tool_choice=None,
        response_format=None,
        stream=False,
        metadata={"operation": "Converse"},
        extensions={},
    )


def _strict_tool_request() -> CanonicalRequest:
    return CanonicalRequest(
        messages=[CanonicalMessage("user", [CanonicalTextBlock("look up account 123")])],
        system=[],
        source_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        target_model="gpt-test",
        max_output_tokens=12,
        temperature=None,
        top_p=None,
        top_k=None,
        stop_sequences=[],
        tools=[
            CanonicalTool(
                "lookup",
                "Lookup",
                {
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                    "required": ["account_id"],
                    "additionalProperties": False,
                },
                strict=True,
            )
        ],
        tool_choice=CanonicalToolChoice("specific", tool_name="lookup"),
        response_format=None,
        stream=False,
        metadata={"operation": "InvokeModel"},
        extensions={},
    )


def _ok_chat_response(**usage_overrides: object) -> dict[str, object]:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        **usage_overrides,
    }
    return {
        "id": "chatcmpl",
        "model": "target-test",
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _stream_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": "text/event-stream", "retry-after": "0"},
        content=content,
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
    assert "strict" not in payload["tools"][0]["function"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert response.content[0].text == "done"
    assert response.usage == CanonicalUsage(10, 2, 12, reasoning_tokens=0, cached_input_tokens=1)


def test_tool_results_preserve_residual_user_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_chat_response())

    request = _request()
    request.messages = [
        CanonicalMessage(
            "user",
            [
                CanonicalToolResultBlock("call_1", [CanonicalTextBlock("42")]),
                CanonicalToolResultBlock("call_2", [CanonicalTextBlock("ok")]),
                CanonicalTextBlock("Now compare both results"),
                CanonicalImageBlock("image/png", data_base64="aGVsbG8="),
            ],
        )
    ]
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    transport.invoke(request, load_provider_profile("openai"), "gpt-test")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["messages"][1:] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now compare both results"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                },
            ],
        },
    ]


def test_openai_transport_payload_includes_strict_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_chat_response())

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    transport.invoke(_strict_tool_request(), load_provider_profile("openai"), "gpt-test")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["tools"][0]["function"]["strict"] is True
    assert payload["tools"][0]["function"]["parameters"]["additionalProperties"] is False


def test_strict_parameters_rejects_unresolved_strict_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("strict capability validation should run before network")

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        strict_parameters=True,
    )
    profile = load_provider_profile(
        "openai",
        overrides={
            "openai": {
                "model_overrides": [
                    {
                        "model": "gpt-test",
                        "capabilities": {
                            "tools": True,
                        },
                    }
                ]
            }
        },
    )

    with pytest.raises(ValidationException, match="strict_tool_schema"):
        transport.invoke(_strict_tool_request(), profile, "gpt-test")


def test_strict_parameters_allows_model_override_for_strict_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_chat_response())

    profile = load_provider_profile(
        "openai",
        overrides={
            "openai": {
                "model_overrides": [
                    {
                        "model": "gpt-test",
                        "capabilities": {
                            "tools": True,
                            "strict_tool_schema": True,
                        },
                    }
                ]
            }
        },
    )
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        strict_parameters=True,
    )

    transport.invoke(_strict_tool_request(), profile, "gpt-test")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["tools"][0]["function"]["strict"] is True


def test_openrouter_cache_point_maps_to_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_ok_chat_response(
                prompt_tokens_details={"cached_tokens": 8, "cache_write_tokens": 2}
            ),
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    response = transport.invoke(
        _cache_point_request("1h"), load_provider_profile("openrouter"), "or-test"
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    system_content = payload["messages"][0]["content"]
    assert system_content == [
        {
            "type": "text",
            "text": "Shared policy document",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert "prompt_cache_key" not in payload
    assert response.usage.cached_input_tokens == 8
    assert response.usage.cache_write_input_tokens == 2


def test_openai_cache_point_adds_prompt_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_chat_response())

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    transport.invoke(_cache_point_request(), load_provider_profile("openai"), "gpt-test")
    changed_request = _cache_point_request()
    changed_request.messages = [
        CanonicalMessage("user", [CanonicalTextBlock("Different question?")])
    ]
    transport.invoke(changed_request, load_provider_profile("openai"), "gpt-test")

    payload = captured[0]
    assert isinstance(payload, dict)
    assert str(payload["prompt_cache_key"]).startswith("bedmock-cachepoint-")
    assert captured[1]["prompt_cache_key"] == payload["prompt_cache_key"]
    assert payload["messages"][0]["content"] == "Shared policy document"


@pytest.mark.parametrize(
    ("provider_id", "env_name", "env_value"),
    [
        ("gemini", "GEMINI_API_KEY", "gemini-test"),
        ("groq", "GROQ_API_KEY", "groq-test"),
    ],
)
def test_cache_point_is_noop_for_automatic_cache_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    env_name: str,
    env_value: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_chat_response())

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    transport.invoke(_cache_point_request("5m"), load_provider_profile(provider_id), "target-test")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert "prompt_cache_key" not in payload
    assert "cache_control" not in json.dumps(payload)
    assert payload["messages"][0]["content"] == "Shared policy document"


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


def test_openai_transport_malformed_success_is_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not-json"))
        ),
        max_retries=0,
    )

    with pytest.raises(InternalServerException) as exc_info:
        transport.invoke(_request(), load_provider_profile("openai"), "gpt-test")

    assert exc_info.value.response["ResponseMetadata"]["HTTPStatusCode"] == 500


def test_openai_transport_invalid_success_schema_is_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        ),
        max_retries=0,
    )

    with pytest.raises(InternalServerException) as exc_info:
        transport.invoke(_request(), load_provider_profile("openai"), "gpt-test")

    assert exc_info.value.response["ResponseMetadata"]["HTTPStatusCode"] == 500


def test_openai_transport_preserves_provider_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(403, json={"error": {"message": "denied"}})
            )
        ),
        max_retries=0,
    )

    with pytest.raises(AccessDeniedException):
        transport.invoke(_request(), load_provider_profile("openai"), "gpt-test")


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


def test_openai_transport_stream_retries_before_first_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stream_response('{"error":"busy"}', status_code=503)
        return _stream_response(
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )

    events = list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))

    assert calls == 2
    assert events[-1].event_type == "message_stop"
    assert events[-1].finish_reason == "end_turn"


def test_openai_transport_stream_retries_provider_error_before_first_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stream_response('data: {"error":{"message":"busy","status":503}}\n\n')
        return _stream_response(
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )

    events = list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))

    assert calls == 2
    assert events[-1].event_type == "message_stop"


def test_openai_transport_stream_error_after_output_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _stream_response(
            'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            'data: {"error":{"message":"upstream failed","status":503}}\n\n'
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
    )
    events = transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test")

    assert next(events).event_type == "message_start"
    assert next(events).event_type == "content_block_start"
    assert next(events).text_delta == "partial"
    with pytest.raises(ServiceUnavailableException, match="upstream failed"):
        next(events)
    assert calls == 1


def test_openai_transport_stream_rejects_incomplete_or_malformed_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    responses = iter(
        [
            _stream_response(
                'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            ),
            _stream_response("data: nope\n\n"),
        ]
    )
    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: next(responses))),
        max_retries=0,
    )

    with pytest.raises(ModelStreamErrorException, match="without a finish reason"):
        list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))
    with pytest.raises(ModelStreamErrorException, match="malformed SSE JSON"):
        list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))


def test_openai_transport_stream_uses_unique_text_and_tool_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            'data: {"choices":[{"delta":{"content":"checking"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\\"q\\":"}},'
            '{"index":1,"id":"call_2","function":{"name":"other","arguments":"{}"}}'
            ']},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"function":{"arguments":"\\"x\\"}"}}'
            ']},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    events = list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))

    starts = [event for event in events if event.event_type == "content_block_start"]
    stops = [event for event in events if event.event_type == "content_block_stop"]
    assert [(event.content_block_index, event.content_block_type) for event in starts] == [
        (0, "text"),
        (1, "tool_use"),
        (2, "tool_use"),
    ]
    assert [event.content_block_index for event in stops] == [0, 1, 2]
    tool_deltas = [event for event in events if event.tool_arguments_delta is not None]
    assert [(event.content_block_index, event.tool_call_id) for event in tool_deltas] == [
        (1, "call_1"),
        (2, "call_2"),
        (1, "call_1"),
    ]


def test_openai_transport_stream_allocates_text_after_tool_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{}"}}'
            ']},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
        )

    transport = OpenAIChatCompletionsTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    events = list(transport.invoke_stream(_request(), load_provider_profile("openai"), "gpt-test"))

    starts = [event for event in events if event.event_type == "content_block_start"]
    assert [(event.content_block_index, event.content_block_type) for event in starts] == [
        (0, "tool_use"),
        (1, "text"),
    ]
