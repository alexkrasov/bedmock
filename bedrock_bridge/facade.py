"""Public boto3-compatible facade functions."""

from __future__ import annotations

import importlib
from typing import Any

from .client import BedrockRuntimeClient
from .config import load_config


def client(
    service_name: str,
    region_name: str | None = None,
    api_version: str | None = None,
    use_ssl: bool = True,
    verify: bool | str | None = None,
    endpoint_url: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    config: object | None = None,
    **kwargs: Any,
) -> Any:
    bridge_config = load_config()
    if service_name == "bedrock-runtime":
        return BedrockRuntimeClient(
            region_name=region_name,
            api_version=api_version,
            use_ssl=use_ssl,
            verify=verify,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=config,
            bridge_config=bridge_config,
            **kwargs,
        )

    if bridge_config.delegate_other_services:
        boto3 = importlib.import_module("boto3")
        return boto3.client(
            service_name,
            region_name=region_name,
            api_version=api_version,
            use_ssl=use_ssl,
            verify=verify,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=config,
            **kwargs,
        )

    raise NotImplementedError(
        "bedrock_bridge only implements boto3.client('bedrock-runtime'). Set "
        "BEDROCK_BRIDGE_DELEGATE_OTHER_SERVICES=true to delegate other services "
        "to an installed boto3 package."
    )
