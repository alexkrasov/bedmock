from __future__ import annotations

import json
from pathlib import Path

import pytest

from bedmock.config import load_config, load_env_file
from bedmock.exceptions import ConfigurationError
from bedmock.provider_profiles import load_provider_profile
from bedmock.routing import resolve_route


def test_env_config_and_provider_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEDMOCK_PROVIDER", "gemini")
    monkeypatch.setenv("BEDMOCK_MODEL", "gemini-test")
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
                "export BEDMOCK_PROVIDER=gemini",
                "BEDMOCK_MODEL=${TARGET_MODEL}",
                'GEMINI_API_KEY="secret with spaces"',
            ]
        ),
        encoding="utf-8",
    )

    env = load_env_file(env_file, env={"TARGET_MODEL": "gemini-test"})

    assert env["AWS_REGION"] == "us-east-1"
    assert env["BEDROCK_MODEL_ID"] == "us.amazon.nova-2-lite-v1:0"
    assert env["BEDMOCK_PROVIDER"] == "gemini"
    assert env["BEDMOCK_MODEL"] == "gemini-test"
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


def test_bedmock_config_filename_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bedmock.json").write_text(
        json.dumps({"default": {"provider": "gemini", "model": "gemini-model"}}),
        encoding="utf-8",
    )

    config = load_config()

    assert config.provider == "gemini"
    assert config.model == "gemini-model"


def test_example_bedmock_json_loads_and_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    repo_root = Path(__file__).parents[2]
    example = repo_root / "examples" / "bedmock.json"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bedmock.json").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    config = load_config()
    route = resolve_route(config, "anthropic.claude-3-haiku-20240307-v1:0")
    groq_route = resolve_route(config, "bedmock.local.groq.oss-coder")

    assert config.provider == "gemini"
    assert (
        config.provider_overrides["groq"]["parameter_policy"]["fixed_values"]["reasoning_format"]
        == "hidden"
    )
    assert route.route_id == "claude-haiku-to-openai"
    assert route.provider_id == "openai"
    assert route.source_codec == "anthropic_messages"
    assert groq_route.provider_id == "groq"
