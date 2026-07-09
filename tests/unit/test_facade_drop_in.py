from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError

import bedmock as boto3
from bedmock import Session
from bedmock.canonical import CanonicalCachePointBlock
from bedmock.exceptions import BedmockCompatibilityWarning, ValidationException
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


def test_invoke_model_rejects_unknown_parameters_before_transport(
    bedmock_env: None,
    fake_transport: object,
) -> None:
    client = boto3.client("bedrock-runtime")

    with pytest.raises(ValidationException, match=r"unknown parameter.*typo"):
        client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=anthropic_body(),
            typo=True,
        )

    assert cast(Any, fake_transport).requests == []


def test_invoke_model_bedrock_controls_fail_before_transport_by_default(
    bedmock_env: None,
    fake_transport: object,
) -> None:
    client = boto3.client("bedrock-runtime")

    with pytest.raises(ValidationException, match="guardrailIdentifier, guardrailVersion"):
        client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=anthropic_body(),
            guardrailIdentifier="guardrail-id",
            guardrailVersion="1",
        )

    assert cast(Any, fake_transport).requests == []


def test_invoke_model_passthrough_warns_and_preserves_controls(
    bedmock_env: None,
    fake_transport: object,
    tmp_path: Any,
) -> None:
    (tmp_path / "bedmock.json").write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {"bedrock_controls": {"mode": "passthrough"}},
                }
            }
        ),
        encoding="utf-8",
    )
    client = boto3.client("bedrock-runtime")

    with pytest.warns(BedmockCompatibilityWarning) as warning_info:
        client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=anthropic_body(),
            guardrailIdentifier="guardrail-id",
            guardrailVersion="1",
            trace="ENABLED",
        )

    assert len(warning_info) == 1
    assert "guardrailIdentifier, guardrailVersion, trace" in str(warning_info[0].message)
    request = cast(Any, fake_transport).requests[0]
    assert request.extensions["bedrock_controls"] == {
        "guardrailIdentifier": "guardrail-id",
        "guardrailVersion": "1",
        "trace": "ENABLED",
    }


def test_invoke_model_mapped_controls_do_not_warn(
    bedmock_env: None,
    fake_transport: object,
) -> None:
    transport = cast(Any, fake_transport)
    transport.supported_bedrock_controls = frozenset({"trace"})
    client = boto3.client("bedrock-runtime")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=anthropic_body(),
            trace="ENABLED",
        )

    assert caught == []
    assert transport.requests[0].extensions["bedrock_controls"] == {"trace": "ENABLED"}


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
    assert events[1]["contentBlockDelta"]["delta"]["text"] == "bedmock"
    assert all("contentBlockStart" not in event for event in events)
    assert events[-1]["metadata"]["usage"]["totalTokens"] == 5


def test_converse_accepts_cache_point_blocks(
    bedmock_env: None,
    fake_transport: object,
) -> None:
    client = boto3.client("bedrock-runtime")
    response = client.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        system=[
            {"text": "Shared lesson policy."},
            {"cachePoint": {"type": "default", "ttl": "5m"}},
        ],
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
    )

    assert response["output"]["message"]["content"][0]["text"] == "bedmock ok"
    request = fake_transport.requests[0]
    assert request.system[-1] == CanonicalCachePointBlock(ttl="5m")


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
