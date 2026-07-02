# Adding A Provider

OpenAI-compatible providers should be added with a JSON provider profile.

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
