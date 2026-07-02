"""OpenAI-compatible Chat Completions transport."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from bedrock_bridge.canonical import (
    CanonicalContentBlock,
    CanonicalImageBlock,
    CanonicalJsonBlock,
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalResponseFormat,
    CanonicalStreamEvent,
    CanonicalTextBlock,
    CanonicalToolResultBlock,
    CanonicalToolUseBlock,
    CanonicalUsage,
)
from bedrock_bridge.exceptions import (
    AccessDeniedException,
    UnsupportedOperationException,
    ValidationException,
)
from bedrock_bridge.provider_profiles import ProviderProfile

from .http_errors import (
    map_network_error,
    provider_request_id_from_response,
    raise_for_provider_status,
)
from .retry import with_retries
from .sse import iter_sse_json


class OpenAIChatCompletionsTransport:
    transport_id = "openai_chat_completions"

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        connect_timeout_seconds: float = 10.0,
        max_retries: int = 2,
        verify: bool | str | None = None,
        client: httpx.Client | None = None,
        strict_parameters: bool = False,
        debug: bool = False,
    ) -> None:
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=connect_timeout_seconds,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        self._client = client or httpx.Client(
            timeout=timeout, verify=True if verify is None else verify
        )
        self._owns_client = client is None
        self.max_retries = max_retries
        self.strict_parameters = strict_parameters
        self.debug = debug

    def invoke(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> CanonicalResponse:
        payload = self._payload(request, provider, target_model, stream=False)
        headers = self._headers(provider)
        operation_name = str(request.metadata.get("operation", "InvokeModel"))

        def send() -> httpx.Response:
            return self._client.post(provider.endpoint_url, headers=headers, json=payload)

        try:
            response = with_retries(send, max_retries=self.max_retries)
            raise_for_provider_status(response, operation_name)
            data = response.json()
        except Exception as exc:
            if isinstance(exc, ValidationException):
                raise
            mapped = map_network_error(exc, operation_name)
            raise mapped from exc
        if not isinstance(data, dict):
            raise ValidationException("Provider response must be a JSON object")
        request_id = self._provider_request_id(data, response)
        return self._canonical_response(data, request_id=request_id, provider_model=target_model)

    def invoke_stream(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> Iterator[CanonicalStreamEvent]:
        payload = self._payload(request, provider, target_model, stream=True)
        headers = self._headers(provider)
        operation_name = str(request.metadata.get("operation", "InvokeModelWithResponseStream"))

        def stream_events() -> Iterator[CanonicalStreamEvent]:
            sequence = 0
            visible = False
            text_started = False
            open_blocks: set[int] = set()
            finish_reason: str | None = None
            usage: CanonicalUsage | None = None
            try:
                with self._client.stream(
                    "POST",
                    provider.endpoint_url,
                    headers=headers,
                    json=payload,
                ) as response:
                    raise_for_provider_status(response, operation_name)
                    yield CanonicalStreamEvent("message_start", sequence)
                    visible = True
                    sequence += 1
                    for chunk in iter_sse_json(response.iter_lines()):
                        chunk_usage = self._usage_from_provider(chunk.get("usage"))
                        if chunk_usage.total_tokens is not None:
                            usage = chunk_usage
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        if delta.get("content") is not None:
                            if not text_started:
                                yield CanonicalStreamEvent(
                                    "content_block_start",
                                    sequence,
                                    content_block_index=0,
                                    content_block_type="text",
                                )
                                sequence += 1
                                text_started = True
                                open_blocks.add(0)
                            yield CanonicalStreamEvent(
                                "content_block_delta",
                                sequence,
                                content_block_index=0,
                                content_block_type="text",
                                text_delta=str(delta.get("content") or ""),
                            )
                            sequence += 1
                        for tool_call in delta.get("tool_calls") or []:
                            index = int(tool_call.get("index", 0))
                            function = tool_call.get("function") or {}
                            if index not in open_blocks:
                                yield CanonicalStreamEvent(
                                    "content_block_start",
                                    sequence,
                                    content_block_index=index,
                                    content_block_type="tool_use",
                                    tool_call_id=tool_call.get("id"),
                                    tool_name=function.get("name"),
                                )
                                sequence += 1
                                open_blocks.add(index)
                            arguments = function.get("arguments")
                            if arguments:
                                yield CanonicalStreamEvent(
                                    "content_block_delta",
                                    sequence,
                                    content_block_index=index,
                                    content_block_type="tool_use",
                                    tool_call_id=tool_call.get("id"),
                                    tool_name=function.get("name"),
                                    tool_arguments_delta=str(arguments),
                                )
                                sequence += 1
                        if choice.get("finish_reason"):
                            finish_reason = self._finish_reason(choice.get("finish_reason"))
                    for index in sorted(open_blocks):
                        yield CanonicalStreamEvent("content_block_stop", sequence, index)
                        sequence += 1
                    yield CanonicalStreamEvent(
                        "message_delta",
                        sequence,
                        finish_reason=finish_reason or "end_turn",
                        usage=usage,
                    )
                    sequence += 1
                    yield CanonicalStreamEvent(
                        "message_stop",
                        sequence,
                        finish_reason=finish_reason or "end_turn",
                    )
                    sequence += 1
                    if usage:
                        yield CanonicalStreamEvent("metadata", sequence, usage=usage)
            except Exception as exc:
                if visible:
                    raise
                mapped = map_network_error(exc, operation_name)
                raise mapped from exc

        return stream_events()

    def count_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int:
        raise UnsupportedOperationException(
            "Exact token counting is not available for this OpenAI-compatible profile "
            "without a provider-native counting endpoint or tokenizer plugin.",
            operation_name=str(request.metadata.get("operation", "CountTokens")),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self, provider: ProviderProfile) -> dict[str, str]:
        api_key = provider.api_key()
        if not api_key:
            keys = ", ".join(provider.api_key_env)
            raise AccessDeniedException(f"Missing API key. Set one of: {keys}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **provider.headers(),
        }
        return headers

    def _payload(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._messages(request),
        }
        self._add_generation_parameters(payload, request, provider, target_model)
        if stream:
            payload["stream"] = True
        if request.tools:
            self._ensure_capability(request, provider, "tools")
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
        if request.tool_choice:
            payload["tool_choice"] = self._tool_choice(request)
            if request.tool_choice.disable_parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = not request.tool_choice.disable_parallel_tool_calls
        if request.response_format:
            self._ensure_capability(request, provider, "structured_output")
            payload["response_format"] = self._response_format(request.response_format)

        fixed = provider.parameter_policy.get("fixed_values", {})
        if isinstance(fixed, dict):
            payload.update(fixed)
        return {key: value for key, value in payload.items() if value is not None}

    def _messages(self, request: CanonicalRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": self._content(request.system)})
        for message in request.messages:
            messages.extend(self._message(message))
        return messages

    def _message(self, message: CanonicalMessage) -> list[dict[str, Any]]:
        tool_results = [
            block for block in message.content if isinstance(block, CanonicalToolResultBlock)
        ]
        if tool_results:
            return [
                {
                    "role": "tool",
                    "tool_call_id": result.tool_use_id,
                    "content": self._tool_result_content(result.content),
                }
                for result in tool_results
            ]

        tool_uses = [block for block in message.content if isinstance(block, CanonicalToolUseBlock)]
        text_blocks = [
            block for block in message.content if not isinstance(block, CanonicalToolUseBlock)
        ]
        output: dict[str, Any] = {
            "role": message.role,
            "content": self._content(text_blocks) or None,
        }
        if tool_uses:
            output["tool_calls"] = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.arguments
                        if isinstance(block.arguments, str)
                        else json.dumps(block.arguments, ensure_ascii=False),
                    },
                }
                for block in tool_uses
            ]
        return [output]

    def _content(self, blocks: Sequence[CanonicalContentBlock]) -> str | list[dict[str, Any]]:
        if not blocks:
            return ""
        has_image = any(isinstance(block, CanonicalImageBlock) for block in blocks)
        if not has_image:
            return self._tool_result_content(blocks)
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, CanonicalTextBlock):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, CanonicalImageBlock):
                url = block.url
                if block.data_base64:
                    url = f"data:{block.media_type};base64,{block.data_base64}"
                image_url: dict[str, Any] = {"url": url}
                if block.detail:
                    image_url["detail"] = block.detail
                parts.append({"type": "image_url", "image_url": image_url})
            elif isinstance(block, CanonicalJsonBlock):
                parts.append({"type": "text", "text": json.dumps(block.value, ensure_ascii=False)})
            elif isinstance(block, CanonicalReasoningBlock):
                parts.append({"type": "text", "text": block.text})
        return parts

    def _tool_result_content(self, blocks: Sequence[CanonicalContentBlock]) -> str:
        chunks: list[str] = []
        for block in blocks:
            if isinstance(block, CanonicalTextBlock):
                chunks.append(block.text)
            elif isinstance(block, CanonicalJsonBlock):
                chunks.append(json.dumps(block.value, ensure_ascii=False))
            elif isinstance(block, CanonicalReasoningBlock):
                chunks.append(block.text)
        return "".join(chunks)

    def _add_generation_parameters(
        self,
        payload: dict[str, Any],
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> None:
        if request.max_output_tokens is not None:
            payload[self._output_token_parameter(provider, target_model)] = (
                request.max_output_tokens
            )
        if request.temperature is not None:
            temperature = request.temperature
            transforms = provider.parameter_policy.get("transforms", {})
            if transforms.get("zero_temperature_to_epsilon") and temperature == 0:
                temperature = 1e-8
            payload["temperature"] = temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            if self.strict_parameters:
                raise ValidationException("top_k has no universal Chat Completions mapping")
            request.metadata.setdefault("warnings", []).append("top_k omitted by provider policy")
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

    def _output_token_parameter(self, provider: ProviderProfile, target_model: str) -> str:
        policy = provider.output_token_parameter or {"default": "max_tokens"}
        for rule in policy.get("model_rules", []):
            exact = rule.get("model")
            glob = rule.get("model_glob")
            regex = rule.get("model_regex")
            if exact == target_model:
                return str(rule["parameter"])
            if glob and re.fullmatch(str(glob).replace("*", ".*"), target_model):
                return str(rule["parameter"])
            if regex and re.search(str(regex), target_model):
                return str(rule["parameter"])
        return str(policy.get("default", "max_tokens"))

    def _tool_choice(self, request: CanonicalRequest) -> str | dict[str, Any]:
        choice = request.tool_choice
        if choice is None:
            return "auto"
        if choice.mode == "none":
            return "none"
        if choice.mode == "auto":
            return "auto"
        if choice.mode in {"required", "any"}:
            return "required"
        if choice.mode == "specific":
            return {"type": "function", "function": {"name": choice.tool_name}}
        raise ValidationException(f"Unsupported tool choice mode: {choice.mode}")

    def _response_format(self, response_format: CanonicalResponseFormat) -> dict[str, Any]:
        if response_format.mode == "json_object":
            return {"type": "json_object"}
        if response_format.mode == "json_schema":
            json_schema: dict[str, Any] = {
                "name": response_format.name or "response",
                "schema": response_format.schema or {},
            }
            if response_format.strict is not None:
                json_schema["strict"] = response_format.strict
            return {"type": "json_schema", "json_schema": json_schema}
        return {"type": "text"}

    def _ensure_capability(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        capability: str,
    ) -> None:
        value = provider.capabilities.get(capability, "unknown")
        if value is False or (self.strict_parameters and value in {"unknown", "model_dependent"}):
            raise ValidationException(
                f"Provider {provider.id!r} capability {capability!r} is {value!r}; "
                "set an explicit model override or disable strict parameters."
            )
        if value in {"unknown", "model_dependent"}:
            request.metadata.setdefault("warnings", []).append(
                f"Attempting provider/model-dependent capability: {capability}"
            )

    def _canonical_response(
        self,
        data: dict[str, Any],
        *,
        request_id: str | None,
        provider_model: str,
    ) -> CanonicalResponse:
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise ValidationException("Provider response missing choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValidationException("Provider choice must be an object")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ValidationException("Provider message must be an object")
        content: list[CanonicalContentBlock] = []
        text = message.get("content")
        if text:
            content.append(CanonicalTextBlock(str(text)))
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            content.append(
                CanonicalToolUseBlock(
                    id=str(tool_call.get("id") or f"call_{len(content)}"),
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                    provider_id=tool_call.get("id"),
                )
            )
        return CanonicalResponse(
            id=str(data.get("id") or request_id or "chatcmpl_bridge"),
            model=str(data.get("model") or provider_model),
            content=content,
            finish_reason=self._finish_reason(choice.get("finish_reason")),
            usage=self._usage_from_provider(data.get("usage")),
            provider_request_id=request_id or data.get("request_id"),
            raw_provider_response=data if self.debug else None,
            metadata={"provider_object": data.get("object")},
            extensions={},
        )

    def _usage_from_provider(self, usage: Any) -> CanonicalUsage:
        if not isinstance(usage, dict):
            return CanonicalUsage.empty()
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return CanonicalUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            cached_input_tokens=prompt_details.get("cached_tokens"),
        )

    def _provider_request_id(self, data: dict[str, Any], response: httpx.Response) -> str | None:
        return (
            str(data.get("request_id") or provider_request_id_from_response(response) or "") or None
        )

    def _finish_reason(self, reason: Any) -> str | None:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filtered",
        }
        if reason is None:
            return None
        return mapping.get(str(reason), "unknown")
