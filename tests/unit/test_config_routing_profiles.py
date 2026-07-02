from __future__ import annotations

import json

import pytest

from bedrock_bridge.config import load_config, load_env_file
from bedrock_bridge.exceptions import ConfigurationError
from bedrock_bridge.provider_profiles import load_provider_profile
from bedrock_bridge.routing import resolve_route


def test_env_config_and_provider_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEDROCK_BRIDGE_PROVIDER", "gemini")
    monkeypatch.setenv("BEDROCK_BRIDGE_MODEL", "gemini-test")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    config = load_config()
    route = resolve_route(config, "anthropic.claude-3-haiku-20240307-v1:0")
    profile = load_provider_profile(route.provider_id)

    assert route.provider_id == "gemini"
    assert route.target_model == "gemini-test"
    assert profile.endpoint_url.endswith("/chat/completions")
    assert profile.api_key() == "secret"


def test_load_env_file_parses_dotenv_file(tmp_path: object) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comments are ignored",
                "AWS_REGION=us-east-1",
                "BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0",
                "export BEDROCK_BRIDGE_PROVIDER=gemini",
                "BEDROCK_BRIDGE_MODEL=${TARGET_MODEL}",
                'GEMINI_API_KEY="secret with spaces"',
            ]
        ),
        encoding="utf-8",
    )

    env = load_env_file(env_file, env={"TARGET_MODEL": "gemini-test"})

    assert env["AWS_REGION"] == "us-east-1"
    assert env["BEDROCK_MODEL_ID"] == "us.amazon.nova-2-lite-v1:0"
    assert env["BEDROCK_BRIDGE_PROVIDER"] == "gemini"
    assert env["BEDROCK_BRIDGE_MODEL"] == "gemini-test"
    assert env["GEMINI_API_KEY"] == "secret with spaces"


def test_json_route_priority_and_interpolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_MODEL", "gpt-route")
    config_file = tmp_path / "bedmock.json"
    config_file.write_text(
        json.dumps(
            {
                "default": {"provider": "gemini", "model": "gemini-default"},
                "routes": [
                    {
                        "id": "exact",
                        "match": {"model_id": "us.amazon.nova-2-lite-v1:0"},
                        "source_codec": "amazon_nova",
                        "target": {"provider": "openai", "model": "${TARGET_MODEL}"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    route = resolve_route(load_config(), "us.amazon.nova-2-lite-v1:0")
    assert route.route_id == "exact"
    assert route.source_codec == "amazon_nova"
    assert route.target_model == "gpt-route"


def test_conflicting_routes_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bedmock.json").write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "id": "a",
                        "match": {"model_id_glob": "anthropic.*"},
                        "target": {"provider": "openai", "model": "x"},
                    },
                    {
                        "id": "b",
                        "match": {"model_id_glob": "anthropic.*"},
                        "target": {"provider": "gemini", "model": "y"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        resolve_route(load_config(), "anthropic.claude-3-haiku-20240307-v1:0")


def test_legacy_bedrock_bridge_config_filename_still_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bedrock-bridge.json").write_text(
        json.dumps({"default": {"provider": "gemini", "model": "legacy-model"}}),
        encoding="utf-8",
    )

    config = load_config()

    assert config.provider == "gemini"
    assert config.model == "legacy-model"
