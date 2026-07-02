"""Retry helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

RETRYABLE_STATUS = {429, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


def retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def with_retries(fn: Callable[[], T], *, max_retries: int) -> T:
    attempt = 0
    last_response: httpx.Response | None = None
    while True:
        try:
            result = fn()
            if isinstance(result, httpx.Response):
                last_response = result
                if result.status_code in RETRYABLE_STATUS and attempt < max_retries:
                    delay = retry_after_seconds(result)
                    if delay is None:
                        delay = min(2.0, 0.1 * (2**attempt)) + random.uniform(0, 0.05)
                    time.sleep(delay)
                    attempt += 1
                    continue
            return result
        except RETRYABLE_EXCEPTIONS:
            if attempt >= max_retries:
                raise
            delay = retry_after_seconds(last_response)
            if delay is None:
                delay = min(2.0, 0.1 * (2**attempt)) + random.uniform(0, 0.05)
            time.sleep(delay)
            attempt += 1
