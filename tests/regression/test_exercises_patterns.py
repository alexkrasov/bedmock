from __future__ import annotations

import bedrock_bridge as boto3


def test_lesson_02_basic_converse_pattern(bridge_env: None, fake_transport: object) -> None:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "What is 2+2?"}]}],
    )
    assert response["output"]["message"]["content"][0]["text"] == "bridge ok"


def test_lesson_04_streaming_loop_pattern(bridge_env: None, fake_transport: object) -> None:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse_stream(
        modelId="us.amazon.nova-2-lite-v1:0",
        system=[{"text": "Summarize calls."}],
        messages=[{"role": "user", "content": [{"text": "Transcript"}]}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0.3},
    )
    text = []
    usage = None
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                text.append(delta["text"])
        elif "metadata" in event:
            usage = event["metadata"].get("usage")
    assert "".join(text) == "bridge ok"
    assert usage["inputTokens"] == 3
