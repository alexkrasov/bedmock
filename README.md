# Bedmock

Bedmock is a Python compatibility facade for code that already uses the supported Bedrock Runtime
shape of `boto3.client("bedrock-runtime")`.

It lets you keep Bedrock-shaped request code while routing inference through provider transports.
The first built-in transport targets OpenAI-compatible chat-completions APIs and ships profiles for
OpenAI, Gemini, OpenRouter, and Groq. The architecture is deliberately modular: new Bedrock source
schemas, provider profiles, and non-OpenAI-compatible transports are separate extension points.

## Status

Bedmock is in an early GitHub-installable release stage. PyPI publication is intentionally not part
of this stage.

## Install From GitHub

```bash
python -m pip install "git+https://github.com/alexkrasov/bedmock.git"
```

Optional AWS delegation support:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
```

## Basic Usage

```python
import bedmock as boto3

client = boto3.client("bedrock-runtime")
```

The rest of the Bedrock Runtime call can stay shaped like Bedrock.

## What Drop-In Means

Bedmock does not reimplement the AWS SDK. It implements the Bedrock Runtime inference surface used
by applications that call:

- `boto3.client("bedrock-runtime")`
- `boto3.Session(...).client("bedrock-runtime")`
- `boto3.session.Session(...).client("bedrock-runtime")`
- `invoke_model`
- `invoke_model_with_response_stream`
- `converse`
- `converse_stream`
- `count_tokens`
- `client.close()`, context manager usage, `client.meta`, and `client.exceptions`

Other AWS services raise `NotImplementedError` unless explicit delegation is enabled:

```bash
export BEDMOCK_DELEGATE_OTHER_SERVICES=true
```

For applications that also use S3, EC2, or other AWS services, install the optional AWS extra so
Bedmock can delegate those clients to the real SDK:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
```

Then either keep the boundary explicit:

```python
import boto3
import bedmock

s3 = boto3.client("s3")
ec2 = boto3.client("ec2")
llm = bedmock.client("bedrock-runtime")
```

Or use the drop-in import with delegation enabled:

```python
import bedmock as boto3

llm = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")
```

## Modular Architecture

Bedmock is built around a small canonical request/response model, so each part of the compatibility
problem stays isolated:

- **Bedrock operation codecs** implement the public runtime methods such as `converse`,
  `converse_stream`, `invoke_model`, and `count_tokens`.
- **Model-family codecs** translate Bedrock-native payload families such as Anthropic Messages,
  Amazon Nova, Titan Text, Meta Llama, Mistral, and generic prompt payloads.
- **Provider profiles** describe provider-specific endpoints, headers, API-key environment
  variables, capability flags, model overrides, and exact token-counting strategies without
  changing Python code.
- **Provider transports** send canonical requests to a provider API. Bedmock ships an
  `openai_chat_completions` transport for OpenAI-compatible APIs and can load additional transports
  from Python entry points.

That means a new OpenAI-compatible provider can usually be added as JSON. A provider with a
different API shape can be added by implementing a transport plugin while reusing the Bedrock
operation layer, source codecs, canonical model, routing, error handling, and streaming machinery.

## Environment Setup

Bedmock uses the `BEDMOCK_*` environment variable prefix.

Gemini:

```bash
export BEDMOCK_PROVIDER=gemini
export BEDMOCK_MODEL="<current-gemini-model>"
export GEMINI_API_KEY="<secret>"
```

OpenAI:

```bash
export BEDMOCK_PROVIDER=openai
export BEDMOCK_MODEL="<current-openai-model>"
export OPENAI_API_KEY="<secret>"
```

OpenRouter:

```bash
export BEDMOCK_PROVIDER=openrouter
export BEDMOCK_MODEL="<provider/model-slug>"
export OPENROUTER_API_KEY="<secret>"
```

Groq:

```bash
export BEDMOCK_PROVIDER=groq
export BEDMOCK_MODEL="<current-groq-model>"
export GROQ_API_KEY="<secret>"
```

Bedmock never loads `.env` automatically. Application code may still do that itself, and CLI
commands can load a file explicitly with `--env-file`:

```bash
bedmock --env-file .env doctor
```

## JSON Config And Routing

Create `bedmock.json` in the working directory or point `BEDMOCK_CONFIG` at a file:

```json
{
  "default": {
    "provider": "gemini",
    "model": "${GEMINI_TARGET_MODEL}"
  },
  "routes": [
    {
      "id": "claude-haiku",
      "match": {"model_id_glob": "anthropic.claude-3-haiku-*"},
      "source_codec": "anthropic_messages",
      "target": {"provider": "openai", "model": "${OPENAI_TARGET_MODEL}"}
    }
  ]
}
```

Route priority is exact `model_id`, glob, regex, default route, then environment fallback. Multiple
matches at the same priority are configuration errors.

For a fuller copyable config with multiple provider routes and `providers` overrides, see
`examples/bedmock.json`. Config files should contain routing and non-secret provider policy only;
keep API keys in environment variables or an explicitly loaded `.env` file.

## Supported Codecs

- Anthropic Messages on Bedrock
- Anthropic legacy completions
- Meta Llama text generation
- Mistral text generation
- Amazon Titan Text
- Amazon Nova messages
- Restricted generic prompt fallback

Every declared codec has request validation, canonical conversion, response conversion, tests, and
reference or contract coverage.

## Supported Operations

| Bedrock method | Bedmock support level | Provider limitations | Fallback behavior |
| --- | --- | --- | --- |
| `invoke_model` | Implemented | Depends on source codec and target provider capabilities | Validation error before network when schema is unsupported |
| `invoke_model_with_response_stream` | Implemented | Requires provider streaming | Capability/provider errors surface as `ClientError` |
| `converse` | Implemented through a separate operation codec | Tool, image, and structured output support is model-dependent | Strict mode rejects unknown capability |
| `converse_stream` | Implemented with ConverseStream taxonomy | Usage appears only if provider emits it | Missing usage remains absent |
| `count_tokens` | Implemented for exact strategies | OpenAI and Gemini have built-in provider-native counters; OpenRouter and Groq do not expose one in Bedmock yet | `UnsupportedOperationException` when no exact strategy exists |

## Tooling And Structured Output

Bedmock maps client-side tool definitions, tool choice, tool calls, tool results, structured output,
and strict tool schema fields where the selected source codec and provider transport support them.
It does not execute tools; application code still runs the tool and returns the result to the
conversation. Provider/model support remains model-dependent for the built-in OpenAI-compatible
profiles.

See `docs/compatibility.md` for the tooling compatibility matrix, including `strict_tool_schema`
capability behavior.

## Prompt Cache Compatibility

Bedrock Converse `cachePoint` blocks are accepted and never sent to the target model as prompt text.
Provider cache behavior depends on the selected profile:

| Provider profile | Bedmock `cachePoint` behavior | Provider cache behavior | Usage mapping |
| --- | --- | --- | --- |
| `openai` | Accepted; omitted from messages; used to derive a stable `prompt_cache_key` | Best effort only; OpenAI prompt caching remains prefix/model dependent and Bedmock does not map Bedrock TTL to OpenAI retention | `cached_tokens` -> `usage.cacheReadInputTokens`; cache writes are not reported by OpenAI chat completions |
| `gemini` | Accepted as a no-op marker | No automatic Bedrock mapping; Bedmock does not create or reference Gemini `cachedContents/...` resources | Cache usage is mapped only if the OpenAI-compatible endpoint returns `prompt_tokens_details.cached_tokens` |
| `openrouter` | Accepted; mapped to `cache_control` on the previous text content part | Explicit prompt caching where the routed model/provider supports it; `ttl: "1h"` is preserved, default/`5m` maps to ephemeral default | `cached_tokens` -> `usage.cacheReadInputTokens`; `cache_write_tokens` -> `usage.cacheWriteInputTokens` |
| `groq` | Accepted as a no-op marker | Groq prompt caching is automatic for eligible repeated prefixes/models; Bedmock sends no cache-specific request fields | `cached_tokens` -> `usage.cacheReadInputTokens`; cache writes are not reported |

## Token Counting

Bedmock implements Bedrock `CountTokens` as exact preflight counting only. It never estimates with
character counts and never runs paid inference just to infer usage.

Built-in exact strategies:

- `openai`: uses OpenAI's Responses input-token counting endpoint.
- `gemini`: uses Gemini's native `models/{model}:countTokens` endpoint.

`openrouter` and `groq` still expose usage from normal inference responses when the provider returns
it, but standalone `client.count_tokens(...)` raises `UnsupportedOperationException` because no
exact preflight counter is configured for those profiles.

## CLI

```bash
bedmock doctor
bedmock list-codecs
bedmock list-providers
bedmock validate-config
bedmock resolve-model "<bedrock-model-id>"
bedmock show-capabilities "<provider-id>"
```

`doctor` does not run paid inference.

## Local Development

```bash
git clone git@github.com:alexkrasov/bedmock.git
cd bedmock
source setup.sh
```

`setup.sh` creates and activates `.venv`, installs Bedmock in editable dev mode, and runs a quick
offline check. For the full local gate:

```bash
BEDMOCK_RUN_CHECKS=full source setup.sh
```

For an app that consumes a local Bedmock checkout:

```bash
cd /path/to/app
source /path/to/bedmock/setup.sh
```

## Documentation

- `docs/configuration.md`
- `examples/bedmock.json`
- `docs/compatibility.md`
- `docs/architecture.md`
- `docs/adding-a-provider.md`
- `docs/adding-a-codec.md`
- `docs/migrating-from-boto3-bedrock.md`

## Security

Bedmock redacts API keys, authorization headers, AWS secrets, prompts, tool arguments, images, and
raw model payloads from logs and exception messages. TLS verification is enabled by default.

## Known Limitations

- Bedrock Guardrails operations are explicit non-goals.
- Async Bedrock jobs are explicit non-goals.
- Bedrock control plane and Agents Runtime are not implemented.
- Token counting requires an exact provider endpoint, official tokenizer, or custom transport
  strategy. Built-in exact strategies currently cover OpenAI and Gemini.
- Provider/model capabilities marked `model_dependent` are attempted only in compatibility mode.
