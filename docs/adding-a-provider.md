# Adding A Provider

Bedmock separates provider identity from provider transport.

If the provider exposes an OpenAI-compatible chat-completions API, add it with a JSON provider
profile. If the provider has a different API shape, implement a custom transport and expose it
through the `bedmock.transports` Python entry-point group.

## OpenAI-Compatible Provider Profile

Required fields:

- `id`
- `transport`
- `base_url`
- `endpoint_path`
- `api_key_env`
- `capabilities`

Optional fields:

- `default_headers`
- `optional_headers`
- `parameter_policy`
- `output_token_parameter`
- `token_counting`
- `model_overrides`

Place custom profiles in a directory and set:

```bash
export BEDMOCK_PROVIDER_PROFILE_PATH=/path/to/provider-profiles
```

Do not store API keys in provider profiles. Use environment variables.

## Exact Token Counting

`count_tokens` must return exact provider counts. Do not wire approximate character or tokenizer
guesses into this path.

For an OpenAI-compatible provider that exposes a Responses-style input-token endpoint, add a
profile strategy:

```json
{
  "token_counting": {
    "strategy": "openai_responses_input_tokens",
    "endpoint_path": "/responses/input_tokens"
  }
}
```

For a Gemini-compatible native counter, use:

```json
{
  "token_counting": {
    "strategy": "gemini_count_tokens",
    "base_url": "https://generativelanguage.googleapis.com/v1beta"
  }
}
```

If a provider has a different exact counting API, implement it in a custom transport rather than
returning estimates.

## Custom Provider Transport

A non-OpenAI-compatible provider needs a transport implementation that satisfies
`bedmock.transports.base.ProviderTransport`.

Register it from your package with the existing entry-point group:

```toml
[project.entry-points."bedmock.transports"]
my_provider = "my_package.transport:MyProviderTransport"
```

Then reference the transport from a provider profile:

```json
{
  "id": "my-provider",
  "transport": "my_provider",
  "base_url": "https://api.provider.example",
  "endpoint_path": "/v1/messages",
  "api_key_env": ["MY_PROVIDER_API_KEY"],
  "capabilities": {}
}
```

The transport receives canonical requests, so it can reuse Bedmock's Bedrock operation handling,
source codecs, routing, validation, streaming envelope handling, and Bedrock-compatible error
surface.
