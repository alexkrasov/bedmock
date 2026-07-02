"""boto3-compatible Session facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .facade import client as create_client


@dataclass
class Session:
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    region_name: str | None = None
    profile_name: str | None = None
    botocore_session: Any | None = None

    def client(
        self,
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
        return create_client(
            service_name,
            region_name=region_name or self.region_name,
            api_version=api_version,
            use_ssl=use_ssl,
            verify=verify,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id or self.aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key or self.aws_secret_access_key,
            aws_session_token=aws_session_token or self.aws_session_token,
            config=config,
            **kwargs,
        )
