"""Incremental Server-Sent Events parser."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from bedmock.exceptions import ValidationException


def iter_sse_data(lines: Iterable[str | bytes]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def iter_sse_json(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    for data in iter_sse_data(lines):
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValidationException(f"Malformed SSE JSON event: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValidationException("SSE JSON event must be an object")
        yield payload
