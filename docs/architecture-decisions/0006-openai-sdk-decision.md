# ADR 0006: OpenAI SDK Decision

## Decision

Core uses raw `httpx` instead of `openai-python`.

## Rationale

The OpenAI SDK is convenient for single-provider application code, but the bridge needs:

- a provider-independent canonical model
- Bedrock request and response conversion
- provider profiles
- custom error mapping
- raw JSON preservation
- a shared retry policy
- support for OpenAI-compatible providers with small differences

The SDK should be reconsidered only as an optional plugin if it can preserve those contracts without
becoming a mandatory dependency.
