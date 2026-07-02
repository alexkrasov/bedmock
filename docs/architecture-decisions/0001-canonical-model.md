# ADR 0001: Canonical Model

## Decision

Use standard-library dataclasses for canonical requests, responses, content blocks, tools, usage,
and stream events.

## Rationale

Dataclasses keep runtime dependencies small and make conversions explicit. Pydantic is useful, but
it is not required for the current validation needs and would add mandatory weight to the core.
