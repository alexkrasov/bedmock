from __future__ import annotations

import bedmock as boto3

client = boto3.client("bedrock-runtime")
response = client.converse(
    modelId="us.amazon.nova-2-lite-v1:0",
    system=[{"text": "Answer briefly."}],
    messages=[{"role": "user", "content": [{"text": "What is 2+2?"}]}],
    inferenceConfig={"maxTokens": 50, "temperature": 0.0},
)
print(response["output"]["message"]["content"][0]["text"])
