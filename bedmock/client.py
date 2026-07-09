"""Bedrock Runtime client facade."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .config import BedmockConfig
from .exceptions import ExceptionsNamespace, UnsupportedOperationException
from .operations import (
    ConverseOperationCodec,
    ConverseStreamOperationCodec,
    CountTokensOperationCodec,
    InvokeModelOperationCodec,
    InvokeModelWithResponseStreamOperationCodec,
)
from .operations.base import OperationContext
from .provider_profiles import ProviderProfile, load_provider_profile
from .routing import resolve_route
from .transports import ProviderTransport, apply_bedrock_control_policy, build_transport


@dataclass
class ServiceModelMetadata:
    service_name: str = "bedrock-runtime"
    operation_names: tuple[str, ...] = (
        "InvokeModel",
        "InvokeModelWithResponseStream",
        "Converse",
        "ConverseStream",
        "CountTokens",
    )


@dataclass
class ClientMeta:
    service_model: ServiceModelMetadata
    region_name: str | None
    endpoint_url: str
    config: object | None


class BedrockRuntimeClient:
    exceptions = ExceptionsNamespace()

    def __init__(
        self,
        *,
        region_name: str | None = None,
        api_version: str | None = None,
        use_ssl: bool = True,
        verify: bool | str | None = None,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        config: object | None = None,
        bedmock_config: BedmockConfig,
        **kwargs: Any,
    ) -> None:
        self.region_name = region_name
        self.api_version = api_version
        self.use_ssl = use_ssl
        self.verify = verify
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.config = config
        self.bedmock_config = bedmock_config
        self.extra_kwargs = kwargs
        self.meta = ClientMeta(
            service_model=ServiceModelMetadata(),
            region_name=region_name,
            endpoint_url=endpoint_url or "bedmock://bedrock-runtime",
            config=config,
        )
        self._transports: dict[str, ProviderTransport] = {}
        self._closed = False

    def __enter__(self) -> BedrockRuntimeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        for transport in self._transports.values():
            transport.close()
        self._closed = True

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        model_id = self._model_id(kwargs)
        context = self._context("InvokeModel", model_id)
        operation = InvokeModelOperationCodec()
        request = operation.decode_operation_request(kwargs, context)
        transport = self._transport(context.provider)
        apply_bedrock_control_policy(request, context.provider, transport)
        response = transport.invoke(
            request,
            context.provider,
            context.route.target_model,
        )
        return operation.encode_operation_response(request, response, context)

    def invoke_model_with_response_stream(self, **kwargs: Any) -> dict[str, Any]:
        model_id = self._model_id(kwargs)
        context = self._context("InvokeModelWithResponseStream", model_id)
        operation = InvokeModelWithResponseStreamOperationCodec()
        request = operation.decode_operation_request(kwargs, context)
        transport = self._transport(context.provider)
        apply_bedrock_control_policy(request, context.provider, transport)
        events = transport.invoke_stream(
            request,
            context.provider,
            context.route.target_model,
        )
        return operation.encode_operation_stream(request, events, context)

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        model_id = self._model_id(kwargs)
        context = self._context("Converse", model_id)
        operation = ConverseOperationCodec()
        request = operation.decode_operation_request(kwargs, context)
        transport = self._transport(context.provider)
        apply_bedrock_control_policy(request, context.provider, transport)
        response = transport.invoke(
            request,
            context.provider,
            context.route.target_model,
        )
        return operation.encode_operation_response(request, response, context)

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        model_id = self._model_id(kwargs)
        context = self._context("ConverseStream", model_id)
        operation = ConverseStreamOperationCodec()
        request = operation.decode_operation_request(kwargs, context)
        transport = self._transport(context.provider)
        apply_bedrock_control_policy(request, context.provider, transport)
        events = transport.invoke_stream(
            request,
            context.provider,
            context.route.target_model,
        )
        return operation.encode_operation_stream(request, events, context)

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        model_id = self._model_id(kwargs)
        context = self._context("CountTokens", model_id)
        operation = CountTokensOperationCodec()
        request = operation.decode_operation_request(kwargs, context)
        transport = self._transport(context.provider)
        apply_bedrock_control_policy(request, context.provider, transport)
        count = transport.count_tokens(
            request,
            context.provider,
            context.route.target_model,
        )
        return operation.encode_count_response(count)

    def apply_guardrail(self, **kwargs: Any) -> None:
        raise UnsupportedOperationException(
            "apply_guardrail is a Bedrock Guardrails operation and has no "
            "provider-independent Bedmock equivalent.",
            operation_name="ApplyGuardrail",
        )

    def start_async_invoke(self, **kwargs: Any) -> None:
        self._raise_async_unsupported("StartAsyncInvoke")

    def get_async_invoke(self, **kwargs: Any) -> None:
        self._raise_async_unsupported("GetAsyncInvoke")

    def list_async_invokes(self, **kwargs: Any) -> None:
        self._raise_async_unsupported("ListAsyncInvokes")

    def _raise_async_unsupported(self, operation_name: str) -> None:
        raise UnsupportedOperationException(
            "Async Bedrock inference jobs require persisted job state and "
            "input/output locations; they are explicit non-goals for bedmock.",
            operation_name=operation_name,
        )

    def _model_id(self, arguments: dict[str, Any]) -> str:
        model_id = arguments.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            from .exceptions import ValidationException

            raise ValidationException("Bedrock Runtime operation requires modelId")
        return model_id

    def _context(self, operation_name: str, model_id: str) -> OperationContext:
        route = resolve_route(self.bedmock_config, model_id)
        provider = load_provider_profile(
            route.provider_id,
            profile_path=self.bedmock_config.provider_profile_path,
            overrides=self.bedmock_config.provider_overrides,
        )
        return OperationContext(
            operation_name=operation_name,
            bedmock_config=self.bedmock_config,
            route=route,
            provider=provider,
            request_id=f"br-{uuid.uuid4().hex}",
        )

    def _transport(self, provider: ProviderProfile) -> ProviderTransport:
        existing = self._transports.get(provider.id)
        if existing is not None:
            return existing
        transport = build_transport(
            provider.transport,
            timeout_seconds=self.bedmock_config.timeout_seconds,
            connect_timeout_seconds=self.bedmock_config.connect_timeout_seconds,
            max_retries=self.bedmock_config.max_retries,
            verify=self.verify,
            strict_parameters=self.bedmock_config.strict_parameters,
            debug=self.bedmock_config.debug,
        )
        self._transports[provider.id] = transport
        return transport
