# ADR 0008: Operation Codecs

## Decision

Bedrock operations are separate codecs from model-family codecs.

## Rationale

Converse is already a standardized Bedrock operation and should not be converted through Anthropic
or another native schema. Native invoke operations use model-family codecs only after the operation
codec has parsed the Bedrock envelope.
