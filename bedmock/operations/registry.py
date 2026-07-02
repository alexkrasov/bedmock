"""Operation registry."""

from __future__ import annotations

from .converse import ConverseOperationCodec
from .converse_stream import ConverseStreamOperationCodec
from .count_tokens import CountTokensOperationCodec
from .invoke_model import InvokeModelOperationCodec
from .invoke_model_stream import InvokeModelWithResponseStreamOperationCodec


def operation_codecs() -> dict[str, object]:
    return {
        "invoke_model": InvokeModelOperationCodec(),
        "invoke_model_with_response_stream": InvokeModelWithResponseStreamOperationCodec(),
        "converse": ConverseOperationCodec(),
        "converse_stream": ConverseStreamOperationCodec(),
        "count_tokens": CountTokensOperationCodec(),
    }
