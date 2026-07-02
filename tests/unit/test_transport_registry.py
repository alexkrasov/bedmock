from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from bedmock.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from bedmock.exceptions import ConfigurationError
from bedmock.provider_profiles import ProviderProfile
from bedmock.transports.base import ProviderTransport
from bedmock.transports.registry import build_transport, list_transport_ids


@dataclass
class FakeEntryPoint:
    name: str
    value: Any

    def load(self) -> Any:
        return self.value


class FakeEntryPoints(list[FakeEntryPoint]):
    def select(self, *, group: str) -> FakeEntryPoints:
        if group == "bedmock.transports":
            return self
        return FakeEntryPoints()


class CustomTransport(ProviderTransport):
    transport_id = "custom_transport"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def invoke(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> CanonicalResponse:
        raise NotImplementedError

    def invoke_stream(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> Iterator[CanonicalStreamEvent]:
        raise NotImplementedError

    def count_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return None


def test_registry_lists_external_transport_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bedmock.transports.registry.metadata.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint("custom_transport", CustomTransport)]),
    )

    assert list_transport_ids() == ["custom_transport", "openai_chat_completions"]


def test_registry_builds_external_transport_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bedmock.transports.registry.metadata.entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint("custom_transport", CustomTransport)]),
    )
    profile = ProviderProfile(
        id="custom",
        transport="custom_transport",
        base_url="https://provider.example",
        endpoint_path="/v1/messages",
        api_key_env=["CUSTOM_API_KEY"],
    )

    transport = build_transport("custom_transport", profile=profile)

    assert isinstance(transport, CustomTransport)
    assert transport.kwargs["profile"] == profile


def test_registry_rejects_ambiguous_external_transports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bedmock.transports.registry.metadata.entry_points",
        lambda: FakeEntryPoints(
            [
                FakeEntryPoint("custom_transport", CustomTransport),
                FakeEntryPoint("custom_transport", CustomTransport),
            ]
        ),
    )

    with pytest.raises(ConfigurationError, match="Ambiguous transport entry point"):
        build_transport("custom_transport")
