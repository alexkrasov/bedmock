from __future__ import annotations


def scenario() -> dict[str, object]:
    return {
        "modelId": "us.amazon.nova-2-lite-v1:0",
        "messages": [{"role": "user", "content": [{"text": "What is 2+2?"}]}],
    }
