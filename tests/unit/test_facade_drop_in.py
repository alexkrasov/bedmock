from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ClientError

import bedmock as boto3
from bedmock import Session
from bedmock.session import Session as ModuleSession
from tests.conftest import anthropic_body


def test_invoke_model_drop_in_streaming_body(bedmock_env: None, fake_transport: object) -> None:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=anthropic_body(),
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(response["body"].read())
    assert payload["content"][0]["text"] == "bedmock ok"
    assert response["contentType"] == "application/json"
    assert response["ResponseMetadata"]["RequestId"] == "req-test"


def test_session_namespaces_create_bedrock_client(
    bedmock_env: None,
    fake_transport: object,
) -> None:
    assert isinstance(Session(region_name="us-east-1").client("bedrock-runtime"), object)
    assert isinstance(ModuleSession(region_name="us-east-1").client("bedrock-runtime"), object)
    assert isinstance(
        boto3.session.Session(region_name="us-east-1").client("bedrock-runtime"), object
    )


def test_converse_stream_matches_bedrock_taxonomy(
    bedmock_env: None, fake_transport: object
) -> None:
    client = boto3.client("bedrock-runtime")
    response = client.converse_stream(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0.0},
    )
    events = list(response["stream"])
    assert "messageStart" in events[0]
    assert events[2]["contentBlockDelta"]["delta"]["text"] == "bedmock"
    assert events[-1]["metadata"]["usage"]["totalTokens"] == 5


def test_count_tokens_uses_transport_exact_strategy(
    bedmock_env: None, fake_transport: object
) -> None:
    client = boto3.client("bedrock-runtime")
    response = client.count_tokens(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        input={
            "invokeModel": {
                "body": anthropic_body(),
                "contentType": "application/json",
            }
        },
    )
    assert response == {"inputTokens": 42}


def test_other_services_require_explicit_delegation(bedmock_env: None) -> None:
    with pytest.raises(NotImplementedError):
        boto3.client("s3")


def test_other_services_delegate_to_installed_boto3(
    bedmock_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_client(service_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"service_name": service_name, "kwargs": kwargs})
        return {"delegated": service_name}

    monkeypatch.setenv("BEDMOCK_DELEGATE_OTHER_SERVICES", "true")
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client))

    response = boto3.client("s3", region_name="us-west-2", endpoint_url="https://s3.example")

    assert response == {"delegated": "s3"}
    assert calls == [
        {
            "service_name": "s3",
            "kwargs": {
                "region_name": "us-west-2",
                "api_version": None,
                "use_ssl": True,
                "verify": None,
                "endpoint_url": "https://s3.example",
                "aws_access_key_id": None,
                "aws_secret_access_key": None,
                "aws_session_token": None,
                "config": None,
            },
        }
    ]


def test_unsupported_non_goals_are_client_errors(bedmock_env: None, fake_transport: object) -> None:
    client = boto3.client("bedrock-runtime")
    with pytest.raises(ClientError) as exc_info:
        client.apply_guardrail()
    assert exc_info.value.response["Error"]["Code"] == "UnsupportedOperationException"
