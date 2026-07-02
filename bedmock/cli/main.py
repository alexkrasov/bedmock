"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from bedmock.codecs import DEFAULT_CODEC_REGISTRY
from bedmock.config import load_config, load_env_file
from bedmock.provider_profiles import list_provider_ids, load_provider_profile
from bedmock.routing import resolve_route
from bedmock.transports import list_transport_ids

from .doctor import doctor, format_doctor


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bedmock")
    parser.add_argument(
        "--env-file",
        action="append",
        dest="env_files",
        help="Load dotenv-style variables from this file for this CLI invocation.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subcommands.add_parser("doctor")
    doctor_parser.add_argument("--model-id", help="Optional source Bedrock modelId to resolve")

    subcommands.add_parser("list-codecs")
    subcommands.add_parser("list-providers")
    subcommands.add_parser("validate-config")

    resolve = subcommands.add_parser("resolve-model")
    resolve.add_argument("model_id")

    capabilities = subcommands.add_parser("show-capabilities")
    capabilities.add_argument("provider_id")

    subcommands.add_parser("list-transports")
    return parser


def _load_cli_env(env_files: list[str] | None) -> dict[str, str] | os._Environ[str]:
    env: dict[str, str] | os._Environ[str] = os.environ
    for env_file in env_files or []:
        env = load_env_file(env_file, env=env)
    return env


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env = _load_cli_env(args.env_files)

    if args.command == "doctor":
        model_id = args.model_id or env.get("BEDROCK_MODEL_ID")
        print(format_doctor(doctor(model_id=model_id, env=env)))
        return 0
    if args.command == "list-codecs":
        print(_json(DEFAULT_CODEC_REGISTRY.names()))
        return 0
    if args.command == "list-providers":
        print(_json(list_provider_ids()))
        return 0
    if args.command == "list-transports":
        print(_json(list_transport_ids()))
        return 0
    if args.command == "validate-config":
        config = load_config(env=env)
        print(_json({"ok": True, "config_path": config.config_path}))
        return 0
    if args.command == "resolve-model":
        config = load_config(env=env)
        print(_json(resolve_route(config, args.model_id)))
        return 0
    if args.command == "show-capabilities":
        config = load_config(env=env)
        profile = load_provider_profile(
            args.provider_id,
            profile_path=config.provider_profile_path,
            overrides=config.provider_overrides,
            env=env,
        )
        print(_json(profile.capabilities))
        return 0
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
