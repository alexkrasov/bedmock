from __future__ import annotations

import bedmock as boto3

client = boto3.client("bedrock-runtime")
response = client.converse_stream(
    modelId="us.amazon.nova-2-lite-v1:0",
    messages=[{"role": "user", "content": [{"text": "Stream one short sentence."}]}],
    inferenceConfig={"maxTokens": 80},
)
for event in response["stream"]:
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]["delta"]
        if "text" in delta:
            print(delta["text"], end="", flush=True)
print()
