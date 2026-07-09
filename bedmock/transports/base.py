"""Provider transport protocol."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, Protocol

from bedmock.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from bedmock.exceptions import BedmockCompatibilityWarning, ValidationException
from bedmock.provider_profiles import ProviderProfile


class ProviderTransport(Protocol):
    transport_id: str

    def invoke(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> CanonicalResponse: ...

    def invoke_stream(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> Iterator[CanonicalStreamEvent]: ...

    def count_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int: ...

    def close(self) -> None: ...


def apply_bedrock_control_policy(
    request: CanonicalRequest,
    provider: ProviderProfile,
    transport: Any,
) -> None:
    """Fail or warn when a transport cannot forward recognized Bedrock controls."""

    controls = request.extensions.get("bedrock_controls")
    if not isinstance(controls, dict) or not controls:
        return
    supported = set(getattr(transport, "supported_bedrock_controls", frozenset()))
    unmapped = sorted(set(controls) - supported)
    if not unmapped:
        return
    operation_name = str(request.metadata.get("operation", "InvokeModel"))
    fields = ", ".join(unmapped)
    mode = provider.bedrock_controls.get("mode", "fail")
    if mode == "fail":
        raise ValidationException(
            f"Provider {provider.id!r} does not support Bedrock control field(s) "
            f"for {operation_name}: {fields}",
            operation_name=operation_name,
        )
    warnings.warn(
        f"Provider {provider.id!r} accepted {operation_name} in passthrough mode but did not "
        f"forward Bedrock control field(s): {fields}",
        BedmockCompatibilityWarning,
        stacklevel=3,
    )
