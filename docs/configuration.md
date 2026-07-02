# Configuration

## Environment Variables

Bedmock uses the `BEDMOCK_*` environment variable prefix.

Core variables:

- `BEDMOCK_PROVIDER`
- `BEDMOCK_MODEL`
- `BEDMOCK_CONFIG`
- `BEDMOCK_TIMEOUT_SECONDS`
- `BEDMOCK_CONNECT_TIMEOUT_SECONDS`
- `BEDMOCK_MAX_RETRIES`
- `BEDMOCK_LOG_LEVEL`
- `BEDMOCK_DEBUG`
- `BEDMOCK_STRICT_PARAMETERS`
- `BEDMOCK_DELEGATE_OTHER_SERVICES`
- `BEDMOCK_PLUGIN_PATH`
- `BEDMOCK_PROVIDER_PROFILE_PATH`
- `BEDMOCK_MAX_IMAGE_BYTES`

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

`BEDMOCK_CONFIG` can point at an explicit JSON config file.

## Env Files For CLI Commands

Bedmock does not load `.env` automatically. To use a local dotenv-style file for one CLI
invocation, pass it explicitly:

```bash
bedmock --env-file .env doctor
bedmock --env-file .env validate-config
bedmock --env-file .env resolve-model "<source-bedrock-model-id>"
```

`--env-file` accepts lines such as `KEY=value`, `export KEY=value`, quoted values, comments, and
`${VAR}` interpolation. Values are loaded only for that command and do not mutate the parent shell.

## Strict Parameters

Set `BEDMOCK_STRICT_PARAMETERS=true` when unsupported or unknown capability mappings should
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
export BEDMOCK_DELEGATE_OTHER_SERVICES=true
```

With delegation enabled, non-Bedrock clients are created by the installed `boto3` package while
`bedrock-runtime` remains routed through Bedmock. Keep secrets in environment variables or the
normal AWS credential chain; do not commit credentials into Bedmock config files.
