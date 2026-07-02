from __future__ import annotations

from bedrock_bridge.config import BridgeConfig
from bedrock_bridge.operations.base import OperationContext
from bedrock_bridge.operations.converse import ConverseOperationCodec
from bedrock_bridge.operations.count_tokens import CountTokensOperationCodec
from bedrock_bridge.operations.invoke_model import InvokeModelOperationCodec
from bedrock_bridge.provider_profiles import load_provider_profile
from bedrock_bridge.routing import RouteResolution

from . import (
    anthropic_messages_basic,
    anthropic_messages_multimodal,
    anthropic_messages_stop_sequences,
    anthropic_messages_system_prompt,
    anthropic_messages_tool_use,
    converse_basic,
    count_tokens_basic,
)


def _context() -> OperationContext:
    return OperationContext(
        operation_name="test",
        bridge_config=BridgeConfig(provider="openai", model="gpt-test"),
        route=RouteResolution("openai", "gpt-test", None, "test"),
        provider=load_provider_profile("openai"),
        request_id="req-test",
    )


def test_reference_invoke_scenarios_decode() -> None:
    codec = InvokeModelOperationCodec()
    for module in [
        anthropic_messages_basic,
        anthropic_messages_system_prompt,
        anthropic_messages_stop_sequences,
        anthropic_messages_tool_use,
        anthropic_messages_multimodal,
    ]:
        request = codec.decode_operation_request(module.scenario(), _context())
        assert request.messages


def test_reference_converse_and_count_tokens_decode() -> None:
    converse = ConverseOperationCodec().decode_operation_request(
        converse_basic.scenario(), _context()
    )
    counted = CountTokensOperationCodec().decode_operation_request(
        count_tokens_basic.count_tokens_scenario(),
        _context(),
    )
    assert converse.messages
    assert counted.messages
