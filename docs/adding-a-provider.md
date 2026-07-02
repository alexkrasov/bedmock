# Adding A Provider

Bedmock separates provider identity from provider transport.

If the provider exposes an OpenAI-compatible chat-completions API, add it with a JSON provider
profile. If the provider has a different API shape, implement a custom transport and expose it
through the `bedrock_bridge.transports` Python entry-point group.

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
- `model_overrides`

Place custom profiles in a directory and set:

```bash
export BEDROCK_BRIDGE_PROVIDER_PROFILE_PATH=/path/to/provider-profiles
```

Do not store API keys in provider profiles. Use environment variables.

## Custom Provider Transport

A non-OpenAI-compatible provider needs a transport implementation that satisfies
`bedrock_bridge.transports.base.ProviderTransport`.

Register it from your package with the existing entry-point group:

```toml
[project.entry-points."bedrock_bridge.transports"]
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
