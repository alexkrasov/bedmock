# Bedmock Provider Boundary Hardening

## Context

Bedmock already has a modular Bedrock-operation, canonical-model, and provider-transport pipeline.
The approved work hardens three boundary failures without changing that architecture:

1. Bedrock control parameters can be accepted even though the selected provider does not support
   or receive them.
2. OpenAI-compatible streaming can misclassify provider failures or complete an incomplete stream
   successfully.
3. Tool-result messages and streamed tool calls can lose content or reuse incompatible content
   block indexes.

## Scope

- Add provider-specific policy for unsupported Bedrock controls through `bedmock.json` provider
  overrides.
- Support `bedrock_controls.mode` values `fail` and `passthrough`, defaulting to `fail`.
- Reject unknown public InvokeModel keyword arguments before network access. Recognized controls
  are exactly `trace`, `guardrailIdentifier`, `guardrailVersion`, `performanceConfigLatency`,
  `requestMetadata`, and `serviceTier`.
- Preserve recognized controls unchanged at `request.extensions["bedrock_controls"]` so transports
  can map them now or in the future.
- In `passthrough` mode, emit an explicit compatibility warning when the built-in transport cannot
  forward one or more accepted controls.
- Correct provider-response and streaming error classification.
- Retry streaming requests only before any provider event becomes visible to the caller.
- Reject provider error events and incomplete streams rather than fabricating a successful
  `end_turn`.
- Preserve residual user content beside tool results.
- Allocate unique canonical indexes for streamed text and tool-use blocks.
- Keep ConverseStream output compatible with Bedrock content-block event rules.
- Add focused regression tests and update configuration/compatibility documentation.

## Out of Scope

- Implementing Bedrock Guardrails locally.
- Translating guardrail identifiers or trace controls to provider-specific moderation APIs.
- Retrying after any model output has been exposed to the caller.
- Replacing the canonical model, operation codec registry, or transport plugin architecture.
- Adding async transports or new providers.

## Assumptions

- `fail` is the safe default when a provider does not support a recognized Bedrock control.
- `passthrough` means the operation accepts and preserves the control for the selected transport;
  it does not claim that the control reached the upstream provider.
- The built-in OpenAI-compatible transport currently maps none of the InvokeModel guardrail,
  trace, performance-latency, or service-tier controls.
- Real HTTP 401/403 responses and missing API keys remain `AccessDeniedException`; malformed
  successful responses and internal protocol failures do not.

## Provider Control Contract

- Provider profiles accept only `bedrock_controls.mode`. Unknown keys or values other than `fail`
  and `passthrough` raise `ConfigurationError` when the profile is loaded.
- An omitted section or omitted mode resolves to `fail`.
- A custom provider profile supplies the base value. A `bedmock.json` provider override replaces
  that section for the selected provider, matching the existing top-level provider override
  precedence.
- A transport may declare `supported_bedrock_controls` as a set of exact canonical field names.
  Declaring a field means the transport is responsible for putting it on the real upstream request.
  Missing declarations are treated as an empty set for plugin compatibility.
- `fail` plus any unmapped control raises `ValidationException` before API-key lookup or network
  I/O. The message includes the provider ID, operation, and sorted field names.
- `passthrough` preserves every control on the canonical request. Mapped fields proceed silently;
  all unmapped fields are dropped by that transport and produce one
  `BedmockCompatibilityWarning` per request listing provider, operation, and sorted dropped fields.
- For a mixed mapped/unmapped request, `fail` rejects the whole request; `passthrough` maps the
  supported subset and warns once for only the unsupported subset.

## Streaming Contract

- Retry is allowed only before the first canonical event is yielded to the caller. HTTP retryable
  statuses, retryable HTTPX startup/read failures, and a retryable provider error in the first SSE
  event may consume the configured retry budget.
- After `message_start` or any later canonical event is yielded, no retry occurs.
- A top-level provider SSE `error` value may be a string or object. Numeric `status` or
  `status_code` values use the existing HTTP-to-Bedmock mapping; otherwise the error becomes
  `ServiceUnavailableException`.
- A stream is complete only after a choice supplies a non-null `finish_reason`. A trailing
  usage-only chunk and `[DONE]` are optional transport framing. EOF without a finish reason becomes
  `ModelStreamErrorException` instead of an invented `end_turn`.
- Malformed successful JSON and unknown internal exceptions become `InternalServerException`.
  Malformed or incomplete streaming protocol after HTTP 200 becomes
  `ModelStreamErrorException`. Explicit 401/403 and missing credentials remain
  `AccessDeniedException`.

## Tool And Event Invariants

- Every provider tool-call index maps to one stable, unique canonical content-block index.
- A text block and tool block never share an index. Text-before-tool, tool-before-text, and multiple
  tool calls each produce matching start/delta/stop lifecycles.
- Canonical text starts remain available to source codecs that need them, while ConverseStream
  suppresses empty text `contentBlockStart` output and emits `contentBlockStart` only for tool use.
- When a canonical user message contains tool results plus residual text or image content, tool
  messages are emitted first and one residual user message preserves the remaining content.

## Implementation Steps

1. Extend provider profiles with validated `bedrock_controls.mode` configuration and document the
   `bedmock.json` override shape.
2. Validate InvokeModel keyword arguments, preserve recognized control fields in request
   extensions, and reject unknown names.
3. At the provider transport boundary, fail or warn according to the selected provider policy.
4. Preserve mixed tool-result and residual user content, and allocate independent canonical stream
   indexes for text and tool calls.
5. Harden streaming startup, error-event detection, completion validation, and pre-output retries.
6. Correct fallback error taxonomy for malformed JSON and unknown internal exceptions.
7. Add regression tests for every reproduced failure and for both provider control modes.
8. Update durable learning notes and the same-day retrospective.

## Verification

- Unknown InvokeModel kwargs fail before any transport request is recorded.
- `fail` mode with any unmapped control raises `ValidationException` with zero network calls.
- `passthrough` mode preserves canonical controls, performs the request, and emits exactly one
  `BedmockCompatibilityWarning` containing the sorted dropped fields; mapped-only requests do not
  warn.
- Provider override precedence, omitted/default mode, invalid mode, and unknown policy keys have
  dedicated tests.
- Stream tests cover retryable `503 -> 200`, a provider error before and after first yield, EOF
  without `finish_reason`, malformed SSE JSON, and no retry after visible output.
- Non-stream tests cover malformed HTTP 200 JSON -> `InternalServerException` and prove 401/403
  remain `AccessDeniedException`.
- Tool tests cover two tool results plus residual text, text then tool, tool then text, and two
  parallel tool calls. Each canonical block has one unique index and one matching stop.
- ConverseStream acceptance asserts the sequence `messageStart`, text delta(s), tool start/delta,
  block stops, `messageStop`, and optional metadata, with no empty text start event.
- Full `pytest` suite.
- `ruff check .` and `ruff format --check .`.
- `mypy bedmock`.
- Package build and metadata check when the local environment supports the existing gate.
- `git diff --check` plus a final review of all visible worktree paths.
