from __future__ import annotations

import json

import bedmock as boto3

client = boto3.client("bedrock-runtime")
response = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "tools": [
                {
                    "name": "lookup_account",
                    "description": "Look up an account by ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {"account_id": {"type": "string"}},
                        "required": ["account_id"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup_account"},
            "messages": [{"role": "user", "content": "Find account 123"}],
        }
    ),
    contentType="application/json",
    accept="application/json",
)
print(json.loads(response["body"].read()))
