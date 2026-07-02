"""Configuration loading for Bedmock."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .exceptions import ConfigurationError

ENV_PREFIX = "BEDMOCK_"
DEFAULT_CONFIG_NAME = "bedmock.json"
ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class RouteTarget:
    provider: str
    model: str


@dataclass
class Route:
    id: str
    match: dict[str, Any]
    target: RouteTarget
    source_codec: str | None = None


@dataclass
class BedmockConfig:
    provider: str | None = None
    model: str | None = None
    config_path: Path | None = None
    routes: list[Route] = field(default_factory=list)
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 2
    log_level: str = "WARNING"
    debug: bool = False
    strict_parameters: bool = False
    delegate_other_services: bool = False
    plugin_path: Path | None = None
    provider_profile_path: Path | None = None
    max_image_bytes: int = 10 * 1024 * 1024
    provider_overrides: dict[str, Any] = field(default_factory=dict)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid float configuration value: {value!r}") from exc


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer configuration value: {value!r}") from exc


def interpolate_env(value: Any, env: dict[str, str] | os._Environ[str] | None = None) -> Any:
    env = env or os.environ
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            return env.get(match.group(1), "")

        return pattern.sub(replace, value)
    if isinstance(value, list):
        return [interpolate_env(item, env) for item in value]
    if isinstance(value, dict):
        return {key: interpolate_env(item, env) for key, item in value.items()}
    return value


def _strip_unquoted_comment(value: str) -> str:
    marker = value.find(" #")
    if marker == -1:
        return value
    return value[:marker].rstrip()


def _parse_env_value(value: str, *, path: Path, line_number: int) -> str:
    value = value.strip()
    if not value:
        return ""
    quote = value[0]
    if quote not in {"'", '"'}:
        return _strip_unquoted_comment(value)
    if len(value) < 2 or value[-1] != quote:
        raise ConfigurationError(f"Unterminated quoted value in env file {path}:{line_number}")
    return value[1:-1]


def load_env_file(
    path: str | Path,
    *,
    env: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, str]:
    """Load a dotenv-style file into a copy of env without mutating os.environ."""

    merged = dict(env or os.environ)
    env_path = Path(path).expanduser()
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Env file not found: {env_path}") from exc

    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"Invalid env file line {env_path}:{line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_PATTERN.fullmatch(key):
            raise ConfigurationError(f"Invalid env var name {key!r} in {env_path}:{line_number}")
        parsed[key] = _parse_env_value(raw_value, path=env_path, line_number=line_number)

    interpolation_env = {**merged, **parsed}
    parsed = {key: str(interpolate_env(value, interpolation_env)) for key, value in parsed.items()}
    return {**merged, **parsed}


def discover_config_path(env: dict[str, str] | os._Environ[str] | None = None) -> Path | None:
    env = env or os.environ
    explicit = env.get("BEDMOCK_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    candidate = Path.cwd() / DEFAULT_CONFIG_NAME
    if candidate.exists():
        return candidate
    return None


def _load_json_config(path: Path, env: dict[str, str] | os._Environ[str]) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Bedmock config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON Bedmock config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Bedmock config root must be a JSON object")
    return cast(dict[str, Any], interpolate_env(raw, env))


def load_config(
    *,
    env: dict[str, str] | os._Environ[str] | None = None,
    explicit_provider: str | None = None,
    explicit_model: str | None = None,
) -> BedmockConfig:
    env = env or os.environ
    config_path = discover_config_path(env)
    raw: dict[str, Any] = {}
    if config_path:
        raw = _load_json_config(config_path, env)

    default = raw.get("default", {})
    if default and not isinstance(default, dict):
        raise ConfigurationError("Bedmock config 'default' must be an object")

    routes: list[Route] = []
    for index, route_raw in enumerate(raw.get("routes", [])):
        if not isinstance(route_raw, dict):
            raise ConfigurationError(f"Route at index {index} must be an object")
        target_raw = route_raw.get("target", {})
        if not isinstance(target_raw, dict):
            raise ConfigurationError(f"Route {route_raw.get('id', index)!r} target must be object")
        provider = target_raw.get("provider")
        model = target_raw.get("model")
        if not provider or not model:
            raise ConfigurationError(
                f"Route {route_raw.get('id', index)!r} needs provider and model"
            )
        routes.append(
            Route(
                id=str(route_raw.get("id") or f"route-{index}"),
                match=dict(route_raw.get("match") or {}),
                target=RouteTarget(provider=str(provider), model=str(model)),
                source_codec=route_raw.get("source_codec"),
            )
        )

    provider = explicit_provider or env.get(f"{ENV_PREFIX}PROVIDER") or default.get("provider")
    model = explicit_model or env.get(f"{ENV_PREFIX}MODEL") or default.get("model")

    profile_path = env.get(f"{ENV_PREFIX}PROVIDER_PROFILE_PATH")
    plugin_path = env.get(f"{ENV_PREFIX}PLUGIN_PATH")
    providers_raw = raw.get("providers", {})
    if providers_raw and not isinstance(providers_raw, dict):
        raise ConfigurationError("Bedmock config 'providers' must be an object")

    return BedmockConfig(
        provider=str(provider) if provider else None,
        model=str(model) if model else None,
        config_path=config_path,
        routes=routes,
        timeout_seconds=_as_float(env.get(f"{ENV_PREFIX}TIMEOUT_SECONDS"), 60.0),
        connect_timeout_seconds=_as_float(env.get(f"{ENV_PREFIX}CONNECT_TIMEOUT_SECONDS"), 10.0),
        max_retries=_as_int(env.get(f"{ENV_PREFIX}MAX_RETRIES"), 2),
        log_level=env.get(f"{ENV_PREFIX}LOG_LEVEL", "WARNING"),
        debug=_as_bool(env.get(f"{ENV_PREFIX}DEBUG"), False),
        strict_parameters=_as_bool(env.get(f"{ENV_PREFIX}STRICT_PARAMETERS"), False),
        delegate_other_services=_as_bool(env.get(f"{ENV_PREFIX}DELEGATE_OTHER_SERVICES"), False),
        plugin_path=Path(plugin_path).expanduser() if plugin_path else None,
        provider_profile_path=Path(profile_path).expanduser() if profile_path else None,
        max_image_bytes=_as_int(env.get(f"{ENV_PREFIX}MAX_IMAGE_BYTES"), 10 * 1024 * 1024),
        provider_overrides=providers_raw,
    )
