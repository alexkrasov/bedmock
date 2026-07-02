from __future__ import annotations

import pytest

from bedmock.exceptions import ValidationException
from bedmock.transports.sse import iter_sse_data, iter_sse_json


def test_sse_parser_handles_comments_fragments_and_done() -> None:
    lines = [
        ": keepalive\n",
        'data: {"a":',
        "data: 1}\n",
        "\n",
        "data: [DONE]\n",
        "\n",
    ]
    assert list(iter_sse_data(lines)) == ['{"a":\n1}', "[DONE]"]
    assert list(iter_sse_json(lines)) == [{"a": 1}]


def test_sse_parser_rejects_malformed_json() -> None:
    with pytest.raises(ValidationException):
        list(iter_sse_json(["data: nope\n", "\n"]))
