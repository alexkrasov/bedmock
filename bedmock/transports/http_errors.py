"""HTTP and network error mapping."""

from __future__ import annotations

from typing import Any

import httpx

from bedmock.exceptions import (
    BedmockError,
    InternalServerException,
    ModelTimeoutException,
    ServiceUnavailableException,
    error_from_http_status,
)
from bedmock.logging_utils import safe_error_message


def provider_request_id_from_response(response: httpx.Response) -> str | None:
    for header in ("x-request-id", "x-openai-request-id", "request-id", "cf-ray"):
        if response.headers.get(header):
            return response.headers[header]
    return None


def extract_provider_error(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except ValueError:
        return safe_error_message(response.text)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or response.reason_phrase
            return safe_error_message(str(message))
        if isinstance(error, str):
            return safe_error_message(error)
        message = payload.get("message")
        if message:
            return safe_error_message(str(message))
    return safe_error_message(response.reason_phrase)


def raise_for_provider_status(response: httpx.Response, operation_name: str) -> None:
    if 200 <= response.status_code < 300:
        return
    exc_type = error_from_http_status(response.status_code)
    raise exc_type(
        extract_provider_error(response),
        operation_name=operation_name,
        request_id=provider_request_id_from_response(response),
        status_code=response.status_code,
    )


def map_network_error(exc: Exception, operation_name: str) -> BedmockError:
    if isinstance(exc, BedmockError):
        return exc
    if isinstance(exc, httpx.ConnectTimeout | httpx.ReadTimeout | httpx.WriteTimeout):
        return ModelTimeoutException(str(exc), operation_name=operation_name)
    if isinstance(exc, httpx.TimeoutException):
        return ModelTimeoutException(str(exc), operation_name=operation_name)
    if isinstance(exc, httpx.ConnectError):
        return ServiceUnavailableException(str(exc), operation_name=operation_name)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        exc_type = error_from_http_status(status)
        return exc_type(
            extract_provider_error(exc.response),
            operation_name=operation_name,
            request_id=provider_request_id_from_response(exc.response),
            status_code=status,
        )
    if isinstance(exc, httpx.HTTPError):
        return ServiceUnavailableException(str(exc), operation_name=operation_name)
    return InternalServerException(str(exc), operation_name=operation_name)
