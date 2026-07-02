"""Doctor command implementation."""

from __future__ import annotations

import os
import platform
from typing import Any

from bedmock.codecs import DEFAULT_CODEC_REGISTRY
from bedmock.config import load_config
from bedmock.provider_profiles import list_provider_ids, load_provider_profile
from bedmock.routing import resolve_route
from bedmock.transports import list_transport_ids
from bedmock.version import __version__


def doctor(
    *,
    model_id: str | None = None,
    env: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    config = load_config(env=env)
    selected_provider = config.provider
    selected_model = config.model
    route_summary: dict[str, Any] | None = None
    configuration_error: str | None = None
    if model_id:
        try:
            route = resolve_route(config, model_id)
        except Exception as exc:
            configuration_error = str(exc)
        else:
            selected_provider = route.provider_id
            selected_model = route.target_model
            route_summary = {
                "route_id": route.route_id,
                "source_codec": route.source_codec,
                "provider": route.provider_id,
                "target_model": route.target_model,
            }

    api_key_found = None
    provider_capabilities: dict[str, Any] | None = None
    if selected_provider:
        try:
            profile = load_provider_profile(
                selected_provider,
                profile_path=config.provider_profile_path,
                overrides=config.provider_overrides,
                env=env,
            )
        except Exception as exc:
            configuration_error = str(exc)
        else:
            api_key_found = bool(profile.api_key(env=env))
            provider_capabilities = profile.capabilities

    return {
        "python_version": platform.python_version(),
        "bedmock_version": __version__,
        "config_path": str(config.config_path) if config.config_path else None,
        "provider": selected_provider,
        "target_model": selected_model,
        "api_key_found": api_key_found,
        "registered_codecs": DEFAULT_CODEC_REGISTRY.names(),
        "registered_transports": list_transport_ids(),
        "registered_providers": list_provider_ids(),
        "matched_route": route_summary,
        "effective_timeout": config.timeout_seconds,
        "connect_timeout": config.connect_timeout_seconds,
        "retries": config.max_retries,
        "strict_parameters": config.strict_parameters,
        "provider_capabilities": provider_capabilities,
        "endpoint_reachability": "not checked without explicit live probe",
        "configuration_error": configuration_error,
    }


def format_doctor(report: dict[str, Any]) -> str:
    lines = ["bedmock doctor"]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
