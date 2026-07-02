# ADR 0005: Runtime Dependencies

## Decision

Mandatory runtime dependencies are `botocore` and `httpx`.

## Rationale

`botocore` provides the AWS-compatible error and streaming body behavior. `httpx` provides TLS,
pooling, timeout control, sync streaming, and proxy support without bringing in provider SDKs.
