from __future__ import annotations

import json


def scenario() -> dict[str, object]:
    return {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "body": json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this small image"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "aGVsbG8=",
                                },
                            },
                        ],
                    }
                ],
            }
        ),
    }
