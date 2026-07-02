from __future__ import annotations

import json

import bedmock as boto3

client = boto3.client("bedrock-runtime")
response = client.count_tokens(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    input={
        "invokeModel": {
            "body": json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 20,
                    "messages": [{"role": "user", "content": "Count this exactly."}],
                }
            ),
            "contentType": "application/json",
        }
    },
)
print(response)
