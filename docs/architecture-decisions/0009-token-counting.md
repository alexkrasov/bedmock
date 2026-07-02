# ADR 0009: Token Counting

## Decision

`count_tokens` returns exact counts only. If no exact provider endpoint, official tokenizer, or
registered plugin is available, it raises `UnsupportedOperationException`.

## Rationale

Approximate token counts would be misleading in cost and quota-sensitive code. The bridge must not
run paid inference only to estimate token counts.
