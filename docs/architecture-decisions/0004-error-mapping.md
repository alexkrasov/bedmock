# ADR 0004: Error Mapping

## Decision

Provider HTTP and network failures map to `botocore.exceptions.ClientError` subclasses exposed
through `client.exceptions`.

## Rationale

Course setup scripts already catch `ClientError`. Mapping errors into Bedrock-style codes preserves
that control flow while keeping provider details redacted.
