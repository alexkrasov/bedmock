# Compatibility

## Boto3 Facade

The following imports work:

```python
import bedmock as boto3
from bedmock import client
from bedmock import Session
from bedmock.session import Session
```

Supported client creation:

```python
boto3.client("bedrock-runtime")
boto3.Session(...).client("bedrock-runtime")
boto3.session.Session(...).client("bedrock-runtime")
```

Common boto3 client parameters are accepted for compatibility. AWS credentials are stored only as
metadata and are not sent to external LLM providers.

## Mixed AWS Applications

For code that uses both Bedrock Runtime inference and other AWS services, prefer an explicit import
boundary:

```python
import boto3
import bedmock

s3 = boto3.client("s3")
ec2 = boto3.client("ec2")
llm = bedmock.client("bedrock-runtime")
```

If the application needs the one-line drop-in import, install the AWS extra and enable delegation:

```bash
python -m pip install "bedmock[aws] @ git+https://github.com/alexkrasov/bedmock.git"
export BEDMOCK_DELEGATE_OTHER_SERVICES=true
```

Then `import bedmock as boto3` keeps `bedrock-runtime` on Bedmock and sends other service clients
to the installed `boto3` package.

## Response Compatibility

`invoke_model` returns a `botocore.response.StreamingBody`, so existing code can call:

```python
payload = json.loads(response["body"].read())
```

Converse responses expose `output.message.content`, `stopReason`, `usage`, and `ResponseMetadata`.
Prompt cache telemetry from OpenAI-compatible providers is mapped to Bedrock usage fields when the
provider returns it: `cached_tokens` becomes `cacheReadInputTokens`, and `cache_write_tokens`
becomes `cacheWriteInputTokens`.

## Prompt Cache Compatibility

Bedrock Converse `cachePoint` blocks are accepted as compatibility markers. Bedmock never includes
the marker itself in prompt text sent to a provider.

| Provider profile | Request handling | Cache behavior | Usage mapping |
| --- | --- | --- | --- |
| `openai` | Drops the marker from messages and derives `prompt_cache_key` from the marked prompt prefix | Best effort. OpenAI decides cache hits from prompt prefixes; Bedrock TTL is not mapped to OpenAI retention. | `cached_tokens` -> `cacheReadInputTokens` |
| `gemini` | Drops the marker from messages | No-op in the built-in OpenAI-compatible Gemini profile. Bedmock does not create or use Gemini `cachedContents/...` resources. | Provider cache usage is mapped only if returned as OpenAI-style `cached_tokens`. |
| `openrouter` | Converts the marker to `cache_control` on the previous text content part | Explicit cache control where the routed model/provider supports it. `ttl: "1h"` is preserved; default/`5m` uses OpenRouter's ephemeral default. | `cached_tokens` -> `cacheReadInputTokens`; `cache_write_tokens` -> `cacheWriteInputTokens` |
| `groq` | Drops the marker from messages | No request field is needed; Groq prompt caching is automatic for eligible repeated prefixes/models. | `cached_tokens` -> `cacheReadInputTokens` |

## CountTokens Compatibility

`count_tokens` returns Bedrock-shaped `{"inputTokens": ...}` responses when the selected provider
profile has an exact preflight counter. Built-in exact support currently covers OpenAI and Gemini.

Providers without an exact configured counter raise `UnsupportedOperationException`; Bedmock does
not return approximate counts. Usage metadata from normal inference responses is still mapped when
providers return it.

## Unsupported Operations

The following are stable diagnostic stubs:

- `apply_guardrail`
- `start_async_invoke`
- `get_async_invoke`
- `list_async_invokes`

They raise `UnsupportedOperationException`, not `AttributeError`.
