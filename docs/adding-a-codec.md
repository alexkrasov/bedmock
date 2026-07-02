# Adding A Codec

Add a model-family codec when a Bedrock native request schema cannot be represented by an existing
codec.

1. Create a module under `bedmock/codecs/`.
2. Implement `can_decode`, `decode_request`, `encode_response`, and `encode_stream_event`.
3. Convert only between the Bedrock schema and canonical dataclasses.
4. Do not read API keys, choose providers, or perform HTTP in the codec.
5. Register the codec in `bedmock/codecs/registry.py` or through the
`bedmock.codecs` entry point group.
6. Add unit tests, golden fixtures, and at least one reference or contract test.

Codec errors must avoid logging full prompts, images, API keys, or tool arguments.
