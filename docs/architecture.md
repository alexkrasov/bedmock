# Architecture

Bedmock keeps three axes separate.

## Operation Codec

Operation codecs implement the public Bedrock Runtime operation:

- `invoke_model`
- `invoke_model_with_response_stream`
- `converse`
- `converse_stream`
- `count_tokens`

They know Bedrock operation envelopes, response metadata, `StreamingBody`, and event stream shapes.
They do not know provider API keys or HTTP endpoints.

## Model-Family Codec

Model-family codecs are used by native invoke operations. They validate and convert Bedrock-native
schemas to and from `CanonicalRequest` and `CanonicalResponse`.

Implemented codecs:

- `anthropic_messages`
- `anthropic_legacy`
- `meta_llama`
- `mistral`
- `amazon_titan_text`
- `amazon_nova`
- `generic_prompt`

## Provider Transport

Provider transports send canonical requests to external model providers. The first transport is
`openai_chat_completions`, implemented with raw `httpx` and JSON/SSE handling.

Additional transports can be registered by installed packages through the
`bedrock_bridge.transports` entry-point group. This keeps non-OpenAI-compatible provider APIs out of
the Bedrock operation and source-codec layers.

The pipeline is:

```text
Bedrock operation
  -> Operation codec
  -> Model-family codec, if native invoke
  -> CanonicalRequest
  -> Provider transport
  -> CanonicalResponse or CanonicalStreamEvent
  -> Operation/model-family codec
  -> Bedrock-compatible response
```

No layer contains a codec-by-provider matrix. Provider differences live in JSON profiles and in the
transport contract.
