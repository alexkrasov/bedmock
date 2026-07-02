from __future__ import annotations

from bedrock_bridge.cli.doctor import doctor
from bedrock_bridge.cli.main import main


def test_doctor_reports_configuration_error_without_crashing(
    monkeypatch: object,
    tmp_path: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    report = doctor(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    assert report["configuration_error"]
    assert report["endpoint_reachability"] == "not checked without explicit live probe"


def test_doctor_cli_loads_explicit_env_file(
    monkeypatch: object,
    tmp_path: object,
    capsys: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0",
                "BEDROCK_BRIDGE_PROVIDER=gemini",
                "BEDROCK_BRIDGE_MODEL=gemini-test",
                "GEMINI_API_KEY=secret",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--env-file", str(env_file), "doctor"]) == 0

    output = capsys.readouterr().out
    assert "provider: gemini" in output
    assert "target_model: gemini-test" in output
    assert "api_key_found: True" in output
    assert "route_id': 'environment'" in output
