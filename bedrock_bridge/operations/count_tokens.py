"""CountTokens operation codec."""

from __future__ import annotations

from typing import Any

from bedrock_bridge.canonical import CanonicalRequest, CanonicalResponse
from bedrock_bridge.exceptions import ValidationException

from .base import OperationContext
from .converse import ConverseOperationCodec
from .invoke_model import InvokeModelOperationCodec


class CountTokensOperationCodec:
    operation_name = "CountTokens"

    def __init__(self) -> None:
        self.invoke_codec = InvokeModelOperationCodec()
        self.converse_codec = ConverseOperationCodec()

    def decode_operation_request(
        self,
        arguments: dict[str, Any],
        context: OperationContext,
    ) -> CanonicalRequest:
        model_id = arguments.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            raise ValidationException("count_tokens requires modelId")
        input_payload = arguments.get("input")
        if not isinstance(input_payload, dict):
            raise ValidationException("count_tokens requires input object")
        if "invokeModel" in input_payload:
            invoke = dict(input_payload["invokeModel"])
            invoke.setdefault("modelId", model_id)
            return self.invoke_codec.decode_operation_request(invoke, context)
        if "converse" in input_payload:
            converse = dict(input_payload["converse"])
            converse.setdefault("modelId", model_id)
            return self.converse_codec.decode_operation_request(converse, context)
        raise ValidationException("count_tokens input must contain invokeModel or converse")

    def encode_operation_response(
        self,
        request: CanonicalRequest,
        response: CanonicalResponse,
        context: OperationContext,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def encode_count_response(self, input_tokens: int) -> dict[str, Any]:
        return {"inputTokens": input_tokens}

    def encode_operation_stream(
        self,
        request: CanonicalRequest,
        events: Any,
        context: OperationContext,
    ) -> Any:
        raise NotImplementedError
