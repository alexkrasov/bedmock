# ADR 0003: Botocore StreamingBody

## Decision

`invoke_model` wraps JSON bytes in `botocore.response.StreamingBody`.

## Rationale

Existing Bedrock code commonly calls `response["body"].read()`, `iter_chunks`, or `iter_lines`.
Using Botocore preserves those semantics instead of creating an incompatible local stream class.
