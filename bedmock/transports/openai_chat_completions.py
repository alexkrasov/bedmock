"""OpenAI-compatible Chat Completions transport."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from hashlib import sha256
from itertools import chain
from typing import Any
from urllib.parse import quote

import httpx

from bedmock.canonical import (
    CanonicalCachePointBlock,
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
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    CanonicalToolUseBlock,
    CanonicalUsage,
)
from bedmock.capabilities import resolve_capability
from bedmock.exceptions import (
    AccessDeniedException,
    BedmockError,
    InternalServerException,
    ModelStreamErrorException,
    ServiceUnavailableException,
    UnsupportedOperationException,
    ValidationException,
    error_from_http_status,
)
from bedmock.provider_profiles import ProviderProfile

from .http_errors import (
    map_network_error,
    provider_request_id_from_response,
    raise_for_provider_status,
)
from .retry import (
    RETRYABLE_EXCEPTIONS,
    RETRYABLE_STATUS,
    sleep_before_retry,
    with_retries,
)
from .sse import iter_sse_json


class OpenAIChatCompletionsTransport:
    transport_id = "openai_chat_completions"
    supported_bedrock_controls: frozenset[str] = frozenset()

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
            raise InternalServerException(
                "Provider response must be a JSON object",
                operation_name=operation_name,
                request_id=provider_request_id_from_response(response),
            )
        request_id = self._provider_request_id(data, response)
        try:
            return self._canonical_response(
                data,
                request_id=request_id,
                provider_model=target_model,
                operation_name=operation_name,
            )
        except BedmockError:
            raise
        except Exception as exc:
            mapped = map_network_error(exc, operation_name)
            raise mapped from exc

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
            response: httpx.Response | None = None
            chunks: Iterator[dict[str, Any]] | None = None
            attempt = 0
            while True:
                try:
                    provider_request = self._client.build_request(
                        "POST",
                        provider.endpoint_url,
                        headers=headers,
                        json=payload,
                    )
                    response = self._client.send(provider_request, stream=True)
                    if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                        response.close()
                        sleep_before_retry(response, attempt)
                        response = None
                        attempt += 1
                        continue
                    raise_for_provider_status(response, operation_name)
                    chunk_iterator = iter_sse_json(response.iter_lines())
                    try:
                        first_chunk = next(chunk_iterator)
                    except StopIteration as exc:
                        raise ModelStreamErrorException(
                            "Provider stream ended before the first event",
                            operation_name=operation_name,
                        ) from exc
                    except ValidationException as exc:
                        raise ModelStreamErrorException(
                            "Provider stream returned malformed SSE JSON",
                            operation_name=operation_name,
                        ) from exc
                    stream_error = self._provider_stream_error(first_chunk, operation_name)
                    if stream_error is not None:
                        if (
                            self._is_retryable_stream_error(stream_error)
                            and attempt < self.max_retries
                        ):
                            response.close()
                            sleep_before_retry(response, attempt)
                            response = None
                            attempt += 1
                            continue
                        raise stream_error
                    chunks = chain((first_chunk,), chunk_iterator)
                    break
                except RETRYABLE_EXCEPTIONS as exc:
                    if response is not None:
                        response.close()
                        response = None
                    if attempt < self.max_retries:
                        sleep_before_retry(None, attempt)
                        attempt += 1
                        continue
                    mapped = map_network_error(exc, operation_name)
                    raise mapped from exc
                except Exception:
                    if response is not None:
                        response.close()
                    raise

            assert response is not None
            assert chunks is not None
            sequence = 0
            next_block_index = 0
            text_index: int | None = None
            tool_indexes: dict[int, int] = {}
            tool_metadata: dict[int, dict[str, str | None]] = {}
            open_blocks: set[int] = set()
            finish_reason: str | None = None
            usage: CanonicalUsage | None = None
            try:
                yield CanonicalStreamEvent("message_start", sequence)
                sequence += 1
                for chunk in chunks:
                    stream_error = self._provider_stream_error(chunk, operation_name)
                    if stream_error is not None:
                        raise stream_error
                    chunk_usage = self._usage_from_provider(chunk.get("usage"))
                    if chunk_usage.total_tokens is not None:
                        usage = chunk_usage
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content") is not None:
                        if text_index is None:
                            text_index = next_block_index
                            next_block_index += 1
                            yield CanonicalStreamEvent(
                                "content_block_start",
                                sequence,
                                content_block_index=text_index,
                                content_block_type="text",
                            )
                            sequence += 1
                            open_blocks.add(text_index)
                        yield CanonicalStreamEvent(
                            "content_block_delta",
                            sequence,
                            content_block_index=text_index,
                            content_block_type="text",
                            text_delta=str(delta.get("content") or ""),
                        )
                        sequence += 1
                    for tool_call in delta.get("tool_calls") or []:
                        provider_index = int(tool_call.get("index", 0))
                        function = tool_call.get("function") or {}
                        if provider_index not in tool_indexes:
                            tool_indexes[provider_index] = next_block_index
                            next_block_index += 1
                            tool_metadata[provider_index] = {
                                "id": tool_call.get("id"),
                                "name": function.get("name"),
                            }
                        metadata = tool_metadata[provider_index]
                        if tool_call.get("id"):
                            metadata["id"] = str(tool_call["id"])
                        if function.get("name"):
                            metadata["name"] = str(function["name"])
                        index = tool_indexes[provider_index]
                        if index not in open_blocks:
                            yield CanonicalStreamEvent(
                                "content_block_start",
                                sequence,
                                content_block_index=index,
                                content_block_type="tool_use",
                                tool_call_id=metadata["id"],
                                tool_name=metadata["name"],
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
                                tool_call_id=metadata["id"],
                                tool_name=metadata["name"],
                                tool_arguments_delta=str(arguments),
                            )
                            sequence += 1
                    if choice.get("finish_reason") is not None:
                        finish_reason = self._finish_reason(choice.get("finish_reason"))
                if finish_reason is None:
                    raise ModelStreamErrorException(
                        "Provider stream ended without a finish reason",
                        operation_name=operation_name,
                    )
                for index in sorted(open_blocks):
                    yield CanonicalStreamEvent("content_block_stop", sequence, index)
                    sequence += 1
                yield CanonicalStreamEvent(
                    "message_delta",
                    sequence,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                sequence += 1
                yield CanonicalStreamEvent(
                    "message_stop",
                    sequence,
                    finish_reason=finish_reason,
                )
                sequence += 1
                if usage:
                    yield CanonicalStreamEvent("metadata", sequence, usage=usage)
            except ValidationException as exc:
                raise ModelStreamErrorException(
                    "Provider stream returned malformed SSE JSON",
                    operation_name=operation_name,
                ) from exc
            except BedmockError:
                raise
            except Exception as exc:
                mapped = map_network_error(exc, operation_name)
                raise mapped from exc
            finally:
                response.close()

        return stream_events()

    def _provider_stream_error(
        self,
        chunk: dict[str, Any],
        operation_name: str,
    ) -> BedmockError | None:
        if "error" not in chunk:
            return None
        raw_error = chunk.get("error")
        status_code: int | None = None
        request_id = chunk.get("request_id")
        if isinstance(raw_error, dict):
            message = raw_error.get("message") or raw_error.get("detail") or raw_error.get("code")
            raw_status = raw_error.get("status_code", raw_error.get("status"))
            if isinstance(raw_status, int) and not isinstance(raw_status, bool):
                status_code = raw_status
            elif isinstance(raw_status, str) and raw_status.isdigit():
                status_code = int(raw_status)
        else:
            message = raw_error
        safe_message = str(message or "Provider reported a streaming error")
        if status_code == 424:
            return ModelStreamErrorException(
                safe_message,
                operation_name=operation_name,
                request_id=str(request_id) if request_id else None,
            )
        if status_code is not None:
            exc_type = error_from_http_status(status_code)
            return exc_type(
                safe_message,
                operation_name=operation_name,
                request_id=str(request_id) if request_id else None,
                status_code=status_code,
            )
        return ServiceUnavailableException(
            safe_message,
            operation_name=operation_name,
            request_id=str(request_id) if request_id else None,
        )

    def _is_retryable_stream_error(self, error: BedmockError) -> bool:
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return isinstance(status_code, int) and status_code in RETRYABLE_STATUS

    def count_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int:
        strategy = str(provider.token_counting.get("strategy") or "")
        if strategy == "openai_responses_input_tokens":
            return self._count_openai_responses_input_tokens(request, provider, target_model)
        if strategy == "gemini_count_tokens":
            return self._count_gemini_tokens(request, provider, target_model)
        raise UnsupportedOperationException(
            f"Exact token counting is not configured for provider {provider.id!r}. "
            "Add a provider-native token_counting strategy or custom transport; "
            "approximate counts are intentionally not returned.",
            operation_name=str(request.metadata.get("operation", "CountTokens")),
        )

    def _count_openai_responses_input_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int:
        endpoint_path = str(
            provider.token_counting.get("endpoint_path") or "/responses/input_tokens"
        )
        url = provider.base_url.rstrip("/") + "/" + endpoint_path.lstrip("/")
        data = self._post_json(
            url=url,
            headers=self._headers(provider),
            payload=self._openai_count_tokens_payload(request, target_model),
            operation_name=str(request.metadata.get("operation", "CountTokens")),
        )
        return self._extract_token_count(data, "input_tokens", "inputTokens")

    def _count_gemini_tokens(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> int:
        data = self._post_json(
            url=self._gemini_count_tokens_url(provider, target_model),
            headers=self._gemini_headers(provider),
            payload=self._gemini_count_tokens_payload(request),
            operation_name=str(request.metadata.get("operation", "CountTokens")),
        )
        return self._extract_token_count(data, "totalTokens", "total_tokens")

    def _post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        operation_name: str,
    ) -> dict[str, Any]:
        def send() -> httpx.Response:
            return self._client.post(url, headers=headers, json=payload)

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
            raise InternalServerException(
                "Provider token count response must be a JSON object",
                operation_name=operation_name,
            )
        return data

    def _extract_token_count(self, data: dict[str, Any], *field_names: str) -> int:
        for field_name in field_names:
            value = data.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        fields = ", ".join(field_names)
        raise ValidationException(f"Provider token count response missing integer field: {fields}")

    def _openai_count_tokens_payload(
        self,
        request: CanonicalRequest,
        target_model: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": target_model,
            "input": self._responses_input(request),
        }
        instructions = self._responses_instructions(request)
        if instructions:
            payload["instructions"] = instructions
        if request.tools:
            payload["tools"] = self._responses_tools(request)
        text_format = self._responses_text_format(request.response_format)
        if text_format:
            payload["text"] = text_format
        return payload

    def _responses_input(self, request: CanonicalRequest) -> list[dict[str, Any]]:
        return [self._responses_message(message, request) for message in request.messages]

    def _responses_message(
        self,
        message: CanonicalMessage,
        request: CanonicalRequest,
    ) -> dict[str, Any]:
        if message.role == "tool":
            self._raise_count_tokens_unsupported(
                request,
                "OpenAI Responses token counting for tool-result history is not mapped yet.",
            )
        return {
            "type": "message",
            "role": message.role,
            "content": self._responses_content(message.content, request),
        }

    def _responses_content(
        self,
        blocks: Sequence[CanonicalContentBlock],
        request: CanonicalRequest,
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, CanonicalTextBlock):
                parts.append({"type": "input_text", "text": block.text})
            elif isinstance(block, CanonicalImageBlock):
                image_url = block.url
                if block.data_base64:
                    image_url = f"data:{block.media_type};base64,{block.data_base64}"
                if not image_url:
                    raise ValidationException("OpenAI image input requires image data or URL")
                part: dict[str, Any] = {"type": "input_image", "image_url": image_url}
                if block.detail:
                    part["detail"] = block.detail
                parts.append(part)
            elif isinstance(block, CanonicalJsonBlock):
                parts.append(
                    {"type": "input_text", "text": json.dumps(block.value, ensure_ascii=False)}
                )
            elif isinstance(block, CanonicalReasoningBlock):
                parts.append({"type": "input_text", "text": block.text})
            elif isinstance(block, CanonicalToolUseBlock | CanonicalToolResultBlock):
                self._raise_count_tokens_unsupported(
                    request,
                    "OpenAI Responses token counting for tool-call history is not mapped yet.",
                )
        if not parts:
            parts.append({"type": "input_text", "text": ""})
        return parts

    def _responses_instructions(self, request: CanonicalRequest) -> str:
        for block in request.system:
            if isinstance(
                block,
                CanonicalImageBlock | CanonicalToolUseBlock | CanonicalToolResultBlock,
            ):
                self._raise_count_tokens_unsupported(
                    request,
                    "OpenAI Responses token counting supports text-only system instructions.",
                )
        return self._tool_result_content(request.system)

    def _responses_tools(self, request: CanonicalRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            item: dict[str, Any] = {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            if tool.strict is not None:
                item["strict"] = tool.strict
            tools.append(item)
        return tools

    def _responses_text_format(
        self,
        response_format: CanonicalResponseFormat | None,
    ) -> dict[str, Any] | None:
        if response_format is None or response_format.mode == "text":
            return None
        if response_format.mode == "json_object":
            return {"format": {"type": "json_object"}}
        if response_format.mode == "json_schema":
            json_schema: dict[str, Any] = {
                "type": "json_schema",
                "name": response_format.name or "response",
                "schema": response_format.schema or {},
            }
            if response_format.strict is not None:
                json_schema["strict"] = response_format.strict
            return {"format": json_schema}
        return None

    def _gemini_count_tokens_url(self, provider: ProviderProfile, target_model: str) -> str:
        base_url = str(
            provider.token_counting.get("base_url")
            or provider.base_url.rstrip("/").removesuffix("/openai")
        )
        model_path = (
            target_model if target_model.startswith("models/") else f"models/{target_model}"
        )
        return base_url.rstrip("/") + "/" + quote(model_path, safe="/") + ":countTokens"

    def _gemini_headers(self, provider: ProviderProfile) -> dict[str, str]:
        api_key = provider.api_key()
        if not api_key:
            keys = ", ".join(provider.api_key_env)
            raise AccessDeniedException(f"Missing API key. Set one of: {keys}")
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            **provider.headers(),
        }

    def _gemini_count_tokens_payload(self, request: CanonicalRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [self._gemini_content(message, request) for message in request.messages],
        }
        if request.system:
            payload["systemInstruction"] = {
                "role": "system",
                "parts": self._gemini_parts(request.system, request, allow_images=False),
            }
        if request.tools:
            payload["tools"] = self._gemini_tools(request)
        tool_config = self._gemini_tool_config(request.tool_choice)
        if tool_config:
            payload["toolConfig"] = tool_config
        generation_config = self._gemini_generation_config(request)
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    def _gemini_content(
        self,
        message: CanonicalMessage,
        request: CanonicalRequest,
    ) -> dict[str, Any]:
        if message.role == "user":
            role = "user"
        elif message.role == "assistant":
            role = "model"
        else:
            self._raise_count_tokens_unsupported(
                request,
                "Gemini token counting for system/tool messages inside conversation history "
                "is not mapped yet.",
            )
        return {
            "role": role,
            "parts": self._gemini_parts(message.content, request, allow_images=True),
        }

    def _gemini_parts(
        self,
        blocks: Sequence[CanonicalContentBlock],
        request: CanonicalRequest,
        *,
        allow_images: bool,
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, CanonicalTextBlock):
                parts.append({"text": block.text})
            elif isinstance(block, CanonicalJsonBlock):
                parts.append({"text": json.dumps(block.value, ensure_ascii=False)})
            elif isinstance(block, CanonicalReasoningBlock):
                parts.append({"text": block.text})
            elif isinstance(block, CanonicalImageBlock):
                if not allow_images:
                    self._raise_count_tokens_unsupported(
                        request,
                        "Gemini token counting supports text-only system instructions.",
                    )
                if block.data_base64:
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": block.media_type,
                                "data": block.data_base64,
                            }
                        }
                    )
                elif block.url:
                    parts.append(
                        {
                            "fileData": {
                                "mimeType": block.media_type,
                                "fileUri": block.url,
                            }
                        }
                    )
                else:
                    raise ValidationException("Gemini image input requires image data or URL")
            elif isinstance(block, CanonicalToolUseBlock | CanonicalToolResultBlock):
                self._raise_count_tokens_unsupported(
                    request,
                    "Gemini token counting for tool-call history is not mapped yet.",
                )
        if not parts:
            parts.append({"text": ""})
        return parts

    def _gemini_tools(self, request: CanonicalRequest) -> list[dict[str, Any]]:
        return [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    }
                    for tool in request.tools
                ]
            }
        ]

    def _gemini_tool_config(
        self,
        tool_choice: CanonicalToolChoice | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        mode_map = {
            "none": "NONE",
            "auto": "AUTO",
            "required": "ANY",
            "any": "ANY",
            "specific": "ANY",
        }
        function_calling_config: dict[str, Any] = {"mode": mode_map[tool_choice.mode]}
        if tool_choice.mode == "specific" and tool_choice.tool_name:
            function_calling_config["allowedFunctionNames"] = [tool_choice.tool_name]
        return {"functionCallingConfig": function_calling_config}

    def _gemini_generation_config(self, request: CanonicalRequest) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            config["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["topP"] = request.top_p
        if request.top_k is not None:
            config["topK"] = request.top_k
        if request.stop_sequences:
            config["stopSequences"] = request.stop_sequences
        if request.response_format:
            if request.response_format.mode == "json_object":
                config["responseMimeType"] = "application/json"
            elif request.response_format.mode == "json_schema":
                config["responseMimeType"] = "application/json"
                config["responseSchema"] = request.response_format.schema or {}
        return config

    def _raise_count_tokens_unsupported(
        self,
        request: CanonicalRequest,
        message: str,
    ) -> None:
        raise UnsupportedOperationException(
            message,
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
            "messages": self._messages(request, provider),
        }
        self._add_generation_parameters(payload, request, provider, target_model)
        prompt_cache_key = self._prompt_cache_key(request, provider, target_model)
        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if stream:
            payload["stream"] = True
        if request.tools:
            self._ensure_capability(request, provider, target_model, "tools")
            if any(tool.strict is not None for tool in request.tools):
                self._ensure_capability(
                    request,
                    provider,
                    target_model,
                    "strict_tool_schema",
                )
            payload["tools"] = [self._chat_completion_tool(tool) for tool in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = self._tool_choice(request)
            if request.tool_choice.disable_parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = not request.tool_choice.disable_parallel_tool_calls
        if request.response_format:
            self._ensure_capability(request, provider, target_model, "structured_output")
            payload["response_format"] = self._response_format(request.response_format)

        fixed = provider.parameter_policy.get("fixed_values", {})
        if isinstance(fixed, dict):
            payload.update(fixed)
        return {key: value for key, value in payload.items() if value is not None}

    def _messages(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": self._content(request.system, provider)})
        for message in request.messages:
            messages.extend(self._message(message, provider))
        return messages

    def _message(
        self,
        message: CanonicalMessage,
        provider: ProviderProfile,
    ) -> list[dict[str, Any]]:
        tool_results = [
            block for block in message.content if isinstance(block, CanonicalToolResultBlock)
        ]
        if tool_results:
            output_messages = [
                {
                    "role": "tool",
                    "tool_call_id": result.tool_use_id,
                    "content": self._tool_result_content(result.content),
                }
                for result in tool_results
            ]
            residual: list[CanonicalContentBlock] = [
                block
                for block in message.content
                if not isinstance(block, CanonicalToolResultBlock)
            ]
            if residual:
                residual_message = CanonicalMessage(message.role, residual)
                output_messages.extend(self._message(residual_message, provider))
            return output_messages

        tool_uses = [block for block in message.content if isinstance(block, CanonicalToolUseBlock)]
        text_blocks = [
            block for block in message.content if not isinstance(block, CanonicalToolUseBlock)
        ]
        output: dict[str, Any] = {
            "role": message.role,
            "content": self._content(text_blocks, provider) or None,
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

    def _content(
        self,
        blocks: Sequence[CanonicalContentBlock],
        provider: ProviderProfile,
    ) -> str | list[dict[str, Any]]:
        if not blocks:
            return ""
        has_image = any(isinstance(block, CanonicalImageBlock) for block in blocks)
        has_openrouter_cache = provider.id == "openrouter" and any(
            isinstance(block, CanonicalCachePointBlock) for block in blocks
        )
        if not has_image and not has_openrouter_cache:
            return self._tool_result_content(blocks)
        parts: list[dict[str, Any]] = []
        last_text_part: dict[str, Any] | None = None
        for block in blocks:
            if isinstance(block, CanonicalTextBlock):
                last_text_part = {"type": "text", "text": block.text}
                parts.append(last_text_part)
            elif isinstance(block, CanonicalImageBlock):
                url = block.url
                if block.data_base64:
                    url = f"data:{block.media_type};base64,{block.data_base64}"
                image_url: dict[str, Any] = {"url": url}
                if block.detail:
                    image_url["detail"] = block.detail
                parts.append({"type": "image_url", "image_url": image_url})
                last_text_part = None
            elif isinstance(block, CanonicalJsonBlock):
                last_text_part = {
                    "type": "text",
                    "text": json.dumps(block.value, ensure_ascii=False),
                }
                parts.append(last_text_part)
            elif isinstance(block, CanonicalReasoningBlock):
                last_text_part = {"type": "text", "text": block.text}
                parts.append(last_text_part)
            elif isinstance(block, CanonicalCachePointBlock) and provider.id == "openrouter":
                if last_text_part is None:
                    continue
                cache_control: dict[str, str] = {"type": "ephemeral"}
                if block.ttl == "1h":
                    cache_control["ttl"] = "1h"
                last_text_part["cache_control"] = cache_control
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

    def _prompt_cache_key(
        self,
        request: CanonicalRequest,
        provider: ProviderProfile,
        target_model: str,
    ) -> str | None:
        if provider.id != "openai":
            return None
        chunks: list[str] = []
        cache_prefix: list[str] | None = None
        for block in request.system:
            if self._append_cache_key_block(chunks, block):
                cache_prefix = list(chunks)
        for message in request.messages:
            chunks.append(f"\nrole:{message.role}\n")
            for block in message.content:
                if self._append_cache_key_block(chunks, block):
                    cache_prefix = list(chunks)
        if cache_prefix is None or not any(chunk.strip() for chunk in cache_prefix):
            return None
        digest = sha256(
            f"{request.source_model_id}\n{target_model}\n{''.join(cache_prefix)}".encode()
        ).hexdigest()
        return f"bedmock-cachepoint-{digest[:32]}"

    def _append_cache_key_block(
        self,
        chunks: list[str],
        block: CanonicalContentBlock,
    ) -> bool:
        if isinstance(block, CanonicalCachePointBlock):
            return True
        if isinstance(block, CanonicalTextBlock):
            chunks.append(block.text)
        elif isinstance(block, CanonicalJsonBlock):
            chunks.append(json.dumps(block.value, sort_keys=True, ensure_ascii=False))
        elif isinstance(block, CanonicalReasoningBlock):
            chunks.append(block.text)
        elif isinstance(block, CanonicalImageBlock):
            chunks.append(block.url or block.data_base64 or "")
        return False

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

    def _chat_completion_tool(self, tool: CanonicalTool) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}

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
        target_model: str,
        capability: str,
    ) -> None:
        value = resolve_capability(
            provider.capabilities,
            capability,
            model=target_model,
            model_overrides=provider.model_overrides,
        )
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
        operation_name: str,
    ) -> CanonicalResponse:
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise InternalServerException(
                "Provider response is missing choices",
                operation_name=operation_name,
                request_id=request_id,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise InternalServerException(
                "Provider choice must be an object",
                operation_name=operation_name,
                request_id=request_id,
            )
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise InternalServerException(
                "Provider message must be an object",
                operation_name=operation_name,
                request_id=request_id,
            )
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
            id=str(data.get("id") or request_id or "chatcmpl_bedmock"),
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
            cache_write_input_tokens=prompt_details.get("cache_write_tokens"),
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
