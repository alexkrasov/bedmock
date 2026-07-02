from __future__ import annotations

import json

import bedmock as boto3

client = boto3.client("bedrock-runtime")
response = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 120,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
            "messages": [{"role": "user", "content": "Return an answer field."}],
        }
    ),
    contentType="application/json",
    accept="application/json",
)
print(json.loads(response["body"].read()))
