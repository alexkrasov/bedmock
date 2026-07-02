from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from typing import Any

import pytest

from bedmock.canonical import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalTextBlock,
    CanonicalUsage,
)


class FakeTransport:
    transport_id = "fake"

    def __init__(self) -> None:
        self.requests: list[CanonicalRequest] = []
        self.closed = False

    def invoke(
        self, request: CanonicalRequest, provider: Any, target_model: str
    ) -> CanonicalResponse:
        self.requests.append(request)
        return CanonicalResponse(
            id="chatcmpl-test",
            model=target_model,
            content=[CanonicalTextBlock("bedmock ok")],
            finish_reason="end_turn",
            usage=CanonicalUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            provider_request_id="req-test",
        )

    def invoke_stream(
        self,
        request: CanonicalRequest,
        provider: Any,
        target_model: str,
    ) -> Iterator[CanonicalStreamEvent]:
        self.requests.append(request)
        yield CanonicalStreamEvent("message_start", 0)
        yield CanonicalStreamEvent("content_block_start", 1, 0, "text")
        yield CanonicalStreamEvent("content_block_delta", 2, 0, "text", text_delta="bedmock")
        yield CanonicalStreamEvent("content_block_delta", 3, 0, "text", text_delta=" ok")
        yield CanonicalStreamEvent("content_block_stop", 4, 0)
        usage = CanonicalUsage(input_tokens=3, output_tokens=2, total_tokens=5)
        yield CanonicalStreamEvent("message_delta", 5, finish_reason="end_turn", usage=usage)
        yield CanonicalStreamEvent("message_stop", 6, finish_reason="end_turn")
        yield CanonicalStreamEvent("metadata", 7, usage=usage)

    def count_tokens(self, request: CanonicalRequest, provider: Any, target_model: str) -> int:
        self.requests.append(request)
        return 42

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def bedmock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEDMOCK_PROVIDER", "openai")
    monkeypatch.setenv("BEDMOCK_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture()
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    transport = FakeTransport()

    def build_transport(*args: Any, **kwargs: Any) -> FakeTransport:
        return transport

    client_module = importlib.import_module("bedmock.client")
    monkeypatch.setattr(client_module, "build_transport", build_transport)
    return transport


def anthropic_body(**overrides: Any) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Say ok"}],
    }
    payload.update(overrides)
    return json.dumps(payload)
