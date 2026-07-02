"""Logging and redaction helpers."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

LOGGER_NAME = "bedrock_bridge"

SENSITIVE_KEYS = {
    "authorization",
    "proxy-authorization",
    "api-key",
    "x-api-key",
    "openai_api_key",
    "gemini_api_key",
    "google_api_key",
    "openrouter_api_key",
    "groq_api_key",
    "aws_secret_access_key",
    "aws_session_token",
    "secret_access_key",
    "session_token",
}

SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(AIza[0-9A-Za-z_\-]{8,})"),
]


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def redact_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: match.group(1) + "<redacted>", redacted)
    return redacted


def redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in mapping.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in SENSITIVE_KEYS:
            redacted[key] = "<redacted>"
        elif isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_mapping(item) if isinstance(item, Mapping) else redact_value(item)
                for item in value
            ]
        else:
            redacted[key] = redact_value(value)
    return redacted


def safe_error_message(message: str, limit: int = 600) -> str:
    message = str(redact_value(message))
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."
