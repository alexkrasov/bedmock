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

Use a config file when the application should route different Bedrock source model IDs to different
providers or when a built-in provider profile needs local capability or parameter-policy overrides:

```json
{
  "default": {
    "provider": "gemini",
    "model": "gemini-default-model"
  },
  "routes": [
    {
      "id": "claude-haiku-to-openai",
      "match": {"model_id_glob": "anthropic.claude-3-haiku-*"},
      "source_codec": "anthropic_messages",
      "target": {"provider": "openai", "model": "openai-tool-model"}
    }
  ],
  "providers": {
    "groq": {
      "parameter_policy": {
        "omit_null_values": true,
        "unsupported": ["logprobs", "logit_bias", "top_logprobs", "messages[].name"],
        "fixed_values": {"n": 1, "reasoning_format": "hidden"},
        "transforms": {"zero_temperature_to_epsilon": true}
      }
    }
  }
}
```

The `routes` array is evaluated by exact `model_id`, then `model_id_glob`, then `model_id_regex`.
The `providers` object overrides built-in or custom provider profiles after they are loaded; nested
objects such as `parameter_policy` are replaced as complete sections, so include the full local
policy you want Bedmock to use.

## Unsupported Bedrock Controls

Some Bedrock Runtime operation fields describe AWS-specific controls rather than model input. For
`invoke_model` and `invoke_model_with_response_stream`, Bedmock recognizes `trace`,
`guardrailIdentifier`, `guardrailVersion`, `performanceConfigLatency`, `requestMetadata`, and
`serviceTier`.

Choose the behavior per provider in `bedmock.json`:

```json
{
  "providers": {
    "openai": {
      "bedrock_controls": {"mode": "fail"}
    },
    "openrouter": {
      "bedrock_controls": {"mode": "passthrough"}
    }
  }
}
```

- `fail` is the default. Any control not mapped by the selected transport raises
  `ValidationException` before API-key lookup or network I/O.
- `passthrough` keeps the control fields in the canonical request so a transport plugin can map
  them. If the selected transport does not forward some fields, the request proceeds and Bedmock
  emits one `BedmockCompatibilityWarning` listing the dropped fields.

The built-in `openai_chat_completions` transport currently maps none of these AWS-specific
controls. Use `passthrough` only when accepting an explicit warning and provider-side omission is
appropriate. Bedmock does not implement AWS Guardrails locally and never silently claims that an
unmapped guardrail was applied.

See `examples/bedmock.json` for a larger copyable example with multiple routes and provider
overrides. The example contains no secrets; put API keys in environment variables such as
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and `GROQ_API_KEY`.

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

Strict tool schema requests are controlled separately from normal tool-use requests. If a request
includes `strict: true` or `strict: false` on a tool definition, strict-parameter mode requires the
selected provider profile or model override to support the `strict_tool_schema` capability.

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
