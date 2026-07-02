"""Provider profile loader."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from bedrock_bridge.config import interpolate_env
from bedrock_bridge.exceptions import ConfigurationError

BUILT_IN_PROVIDERS = ("openai", "gemini", "openrouter", "groq")


@dataclass
class ProviderProfile:
    id: str
    transport: str
    base_url: str
    endpoint_path: str
    api_key_env: list[str]
    default_headers: dict[str, str] = field(default_factory=dict)
    optional_headers: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    parameter_policy: dict[str, Any] = field(default_factory=dict)
    output_token_parameter: dict[str, Any] = field(default_factory=dict)
    token_counting: dict[str, Any] = field(default_factory=dict)
    model_overrides: list[dict[str, Any]] = field(default_factory=list)

    @property
    def endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + "/" + self.endpoint_path.lstrip("/")

    def api_key(self, env: dict[str, str] | os._Environ[str] | None = None) -> str | None:
        env = env or os.environ
        for key in self.api_key_env:
            value = env.get(key)
            if value:
                return value
        return None

    def headers(self, env: dict[str, str] | os._Environ[str] | None = None) -> dict[str, str]:
        env = env or os.environ
        headers = dict(self.default_headers)
        optional = interpolate_env(self.optional_headers, env)
        for key, value in optional.items():
            if value:
                headers[key] = str(value)
        return headers


def _profile_from_dict(raw: dict[str, Any]) -> ProviderProfile:
    missing = [
        field_name
        for field_name in ("id", "transport", "base_url", "endpoint_path", "api_key_env")
        if field_name not in raw
    ]
    if missing:
        raise ConfigurationError(f"Provider profile missing required fields: {', '.join(missing)}")
    return ProviderProfile(
        id=str(raw["id"]),
        transport=str(raw["transport"]),
        base_url=str(raw["base_url"]),
        endpoint_path=str(raw["endpoint_path"]),
        api_key_env=[str(item) for item in raw["api_key_env"]],
        default_headers={str(k): str(v) for k, v in raw.get("default_headers", {}).items()},
        optional_headers={str(k): str(v) for k, v in raw.get("optional_headers", {}).items()},
        capabilities=dict(raw.get("capabilities", {})),
        parameter_policy=dict(raw.get("parameter_policy", {})),
        output_token_parameter=dict(raw.get("output_token_parameter", {})),
        token_counting=dict(raw.get("token_counting", {})),
        model_overrides=list(raw.get("model_overrides", [])),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Provider profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid provider profile JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Provider profile {path} must be a JSON object")
    return raw


def load_provider_profile(
    provider_id: str,
    *,
    profile_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | os._Environ[str] | None = None,
) -> ProviderProfile:
    env = env or os.environ
    raw: dict[str, Any] | None = None
    if profile_path:
        candidate = profile_path / f"{provider_id}.json" if profile_path.is_dir() else profile_path
        if candidate.exists():
            raw = _read_json(candidate)
    if raw is None:
        if provider_id not in BUILT_IN_PROVIDERS:
            raise ConfigurationError(f"Unknown provider profile: {provider_id}")
        resource = resources.files("bedrock_bridge.provider_profiles").joinpath(
            f"{provider_id}.json"
        )
        raw = json.loads(resource.read_text(encoding="utf-8"))

    raw = interpolate_env(raw, env)
    override = (overrides or {}).get(provider_id, {})
    if override:
        raw = {**raw, **override}
        if "capabilities" in override:
            raw["capabilities"] = {
                **dict(raw.get("capabilities", {})),
                **dict(override["capabilities"]),
            }
    return _profile_from_dict(raw)


def list_provider_ids() -> list[str]:
    return list(BUILT_IN_PROVIDERS)
