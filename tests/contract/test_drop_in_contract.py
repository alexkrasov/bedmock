from __future__ import annotations

import json

import bedmock as boto3
from tests.conftest import anthropic_body


def test_application_code_only_changes_import(bedmock_env: None, fake_transport: object) -> None:
    client = boto3.client("bedrock-runtime")
    response = client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=anthropic_body(),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    assert payload["content"][0]["text"] == "bedmock ok"
