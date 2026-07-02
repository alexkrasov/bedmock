# Configuration

## Environment Variables

Bedmock keeps the `BEDROCK_BRIDGE_*` environment variable names for compatibility with existing
applications.

Core variables:

- `BEDROCK_BRIDGE_PROVIDER`
- `BEDROCK_BRIDGE_MODEL`
- `BEDROCK_BRIDGE_CONFIG`
- `BEDROCK_BRIDGE_TIMEOUT_SECONDS`
- `BEDROCK_BRIDGE_CONNECT_TIMEOUT_SECONDS`
- `BEDROCK_BRIDGE_MAX_RETRIES`
- `BEDROCK_BRIDGE_LOG_LEVEL`
- `BEDROCK_BRIDGE_DEBUG`
- `BEDROCK_BRIDGE_STRICT_PARAMETERS`
- `BEDROCK_BRIDGE_DELEGATE_OTHER_SERVICES`
- `BEDROCK_BRIDGE_PLUGIN_PATH`
- `BEDROCK_BRIDGE_PROVIDER_PROFILE_PATH`
- `BEDROCK_BRIDGE_MAX_IMAGE_BYTES`

Provider keys:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`
- `GROQ_API_KEY`

## Config File

The default config filename is `bedmock.json`. Environment interpolation uses `${VAR}`. API keys
should stay in environment variables, not in config files.

`BEDMOCK_CONFIG` can point at an explicit JSON config file. `BEDROCK_BRIDGE_CONFIG` and
`bedrock-bridge.json` remain supported as legacy compatibility paths.

## Env Files For CLI Commands

The bridge does not load `.env` automatically. To use a local dotenv-style file for one CLI
invocation, pass it explicitly:

```bash
bedmock --env-file .env doctor
bedmock --env-file .env validate-config
bedmock --env-file .env resolve-model "<source-bedrock-model-id>"
```

`--env-file` accepts lines such as `KEY=value`, `export KEY=value`, quoted values, comments, and
`${VAR}` interpolation. Values are loaded only for that command and do not mutate the parent shell.

## Strict Parameters

Set `BEDROCK_BRIDGE_STRICT_PARAMETERS=true` when unsupported or unknown capability mappings should
fail before network calls. Compatibility mode can attempt model-dependent capabilities and records a
warning in request metadata, but it never drops system prompts, tools, stop sequences, response
formats, multimodal blocks, or output token limits silently.

## Delegating Other AWS Services

By default, Bedmock only implements `boto3.client("bedrock-runtime")`. Calls such as
`boto3.client("s3")` or `boto3.client("ec2")` raise `NotImplementedError` so applications do not
accidentally assume the facade is a full AWS SDK.

For mixed AWS applications, install the optional AWS dependency and enable delegation:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
export BEDROCK_BRIDGE_DELEGATE_OTHER_SERVICES=true
```

With delegation enabled, non-Bedrock clients are created by the installed `boto3` package while
`bedrock-runtime` remains routed through Bedmock. Keep secrets in environment variables or the
normal AWS credential chain; do not commit credentials into bridge config files.
