from __future__ import annotations

import json


def scenario() -> dict[str, object]:
    return {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "system": "Answer in one sentence.",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "What is DI?"}],
            }
        ),
    }
