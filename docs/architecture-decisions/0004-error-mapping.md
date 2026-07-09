# ADR 0004: Error Mapping

## Decision

Provider HTTP and network failures map to `botocore.exceptions.ClientError` subclasses exposed
through `client.exceptions`.

Explicit HTTP 401/403 and missing provider credentials map to `AccessDeniedException`. Malformed
successful provider JSON and unknown internal failures map to `InternalServerException`, never to
an authorization error. Streaming protocol failures after HTTP 200 map to
`ModelStreamErrorException`; provider error payloads with an explicit status use the normal HTTP
mapping.

## Rationale

Course setup scripts already catch `ClientError`. Mapping errors into Bedrock-style codes preserves
that control flow while keeping provider details redacted.

Accurate classification also preserves retry behavior: callers should not suppress retries because
a provider protocol failure was mislabeled as a permanent authorization error.
