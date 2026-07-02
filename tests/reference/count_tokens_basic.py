from __future__ import annotations

from .anthropic_messages_basic import scenario


def count_tokens_scenario() -> dict[str, object]:
    invoke = scenario()
    return {"modelId": invoke["modelId"], "input": {"invokeModel": invoke}}
