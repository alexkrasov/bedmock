from __future__ import annotations

import json


def scenario() -> dict[str, object]:
    return {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Lookup an account",
                        "input_schema": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "lookup"},
                "messages": [{"role": "user", "content": "Lookup account 123"}],
            }
        ),
    }
