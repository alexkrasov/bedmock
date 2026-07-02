"""Operation codec base types and response helpers."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from botocore.response import StreamingBody

from bedmock.canonical import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from bedmock.config import BedmockConfig
from bedmock.exceptions import ValidationException
from bedmock.provider_profiles import ProviderProfile
from bedmock.routing import RouteResolution


@dataclass
class OperationContext:
    operation_name: str
    bedmock_config: BedmockConfig
    route: RouteResolution
    provider: ProviderProfile
    request_id: str


class OperationCodec(Protocol):
    operation_name: str

    def decode_operation_request(
        self,
        arguments: dict[str, Any],
        context: OperationContext,
    ) -> CanonicalRequest: ...

    def encode_operation_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
        context: OperationContext,
    ) -> dict[str, Any]: ...

    def encode_operation_stream(
        self,
        request: CanonicalRequest,
        events: Iterator[CanonicalStreamEvent],
        context: OperationContext,
    ) -> Any: ...


class BedrockEventStream:
    """Small iterable wrapper with a close method like botocore EventStream."""

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self._events = iter(events)
        self._closed = False

    def __iter__(self) -> BedrockEventStream:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        return next(self._events)

    def close(self) -> None:
        self._closed = True
        close = getattr(self._events, "close", None)
        if callable(close):
            close()


def bedrock_response_metadata(
    *,
    request_id: str,
    status_code: int = 200,
    content_type: str = "application/json",
    retry_attempts: int = 0,
) -> dict[str, Any]:
    return {
        "RequestId": request_id,
        "HTTPStatusCode": status_code,
        "HTTPHeaders": {"content-type": content_type},
        "RetryAttempts": retry_attempts,
    }


def streaming_body_from_json(payload: dict[str, Any]) -> StreamingBody:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return StreamingBody(io.BytesIO(raw), len(raw))


def read_body_as_json(body: Any) -> dict[str, Any]:
    original_position: int | None = None
    if hasattr(body, "tell") and hasattr(body, "seek"):
        try:
            original_position = body.tell()
        except Exception:
            original_position = None

    try:
        if isinstance(body, str):
            raw = body.encode("utf-8")
        elif isinstance(body, bytes | bytearray):
            raw = bytes(body)
        elif hasattr(body, "read"):
            read = body.read()
            raw = read.encode("utf-8") if isinstance(read, str) else bytes(read)
        else:
            raise ValidationException("invoke_model body must be str, bytes, or file-like")
    except ValidationException:
        raise
    except Exception as exc:
        raise ValidationException("Could not read invoke_model body") from exc
    finally:
        if original_position is not None:
            with suppress(Exception):
                body.seek(original_position)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationException("invoke_model body must be UTF-8 JSON") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationException(f"invoke_model body is malformed JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValidationException("invoke_model body JSON must be an object")
    return payload


def json_event(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"chunk": {"bytes": raw}}
