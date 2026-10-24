# tiny-tool-result

> **Structured tool-call result handling for AI agents.** Normalize, validate, truncate, redact, and reason over the heterogeneous outputs of LLM tool calls — JSON, text, images, error envelopes, multi-modal blobs. Zero deps. Single file. MIT.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![Zero deps](https://img.shields.io/badge/dependencies-zero-success.svg)](#)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why?

Every AI agent eventually hits the same wall: tool calls return shapes that don't match the schema, dump 400 KB of JSON into the next prompt, leak `Authorization` headers in error messages, or contain base64 images the model can't actually see. You write a pile of ad-hoc `if isinstance(...)` handlers and live with it.

**tiny-tool-result** replaces the pile with one normalized envelope:

```python
from tiny_tool_result import ToolResult, success, failure, image_result

# Wrap any return value
r = success({"user_id": 42, "name": "ada"}, tool="db.lookup", latency_ms=12)
r = failure("Connection refused", tool="api.fetch", retryable=True, retry_after=2.0)

# Auto-validate against a JSON schema (no jsonschema dependency)
r = success(payload, schema={"type": "object", "required": ["id"]})
if not r.valid:
    print(r.validation_errors)

# Truncate noisy results before they eat the next prompt
r = success(huge_dict, max_bytes=4096)  # auto-truncated with marker

# Redact secrets
r = success(response_dict, redact_keys=["api_key", "Authorization", "*.password"])
print(r.value)  # -> {"Authorization": "[REDACTED]", ...}
```

## Features

- **`ToolResult` envelope** — uniform `{ok, value, error, tool, latency_ms, tokens, ...}` shape regardless of source type
- **Auto-construction helpers** — `success()`, `failure()`, `image_result()`, `text_result()`, `multi_result()`
- **JSON Schema validation** — built-in mini-validator, no `jsonschema` dependency
- **Size limiting** — `max_bytes` auto-truncates with explicit `[TRUNCATED: 12.4KB of 387KB shown]` marker
- **Secret redaction** — recursive `redact_keys=["*.password", "headers.Authorization"]` (glob path syntax)
- **Multi-modal bundling** — `multi_result(text="...", image_b64="...")` packages mixed outputs
- **Token estimation** — `approx_tokens()` via char/4 heuristic; warn if result > N tokens
- **Retry hints** — `failure(..., retryable=True, retry_after=2.0)` propagates to `tiny-retry-plus`
- **Trace correlation** — `trace_id` / `span_id` field, auto-attach from `tiny-trace` if available
- **Async** — every helper has `async` twin (`asuccess`, `afailure`)
- **Zero dependencies** — pure stdlib, ~360 LOC, 38/38 tests passing

## Quick Start

```python
from tiny_tool_result import success, failure, ToolResult

def lookup_user(user_id: int) -> ToolResult:
    try:
        data = db.query("SELECT * FROM users WHERE id=?", user_id)
    except Exception as e:
        return failure(str(e), tool="db.lookup", retryable=True, retry_after=1.0)
    return success(data, tool="db.lookup", latency_ms=elapsed_ms, max_bytes=2048)

r = lookup_user(42)
if r.ok:
    send_to_llm(r.summary())   # truncated, redacted, schema-validated
else:
    if r.retryable:
        retry_queue.put(r)
```

## API Reference

### `ToolResult`
| Field | Type | Description |
|-------|------|-------------|
| `ok` | bool | Success flag |
| `value` | Any | Payload (post-validation, post-truncation) |
| `error` | str | Error message if `ok=False` |
| `tool` | str | Source tool name |
| `latency_ms` | float | Call duration |
| `tokens` | int | Estimated token cost |
| `trace_id` | str | Correlation id |
| `retryable` | bool | Hint to retry layer |
| `retry_after` | float | Seconds hint |
| `truncated` | bool | Whether `max_bytes` cut content |
| `redacted` | list | Keys that were redacted |
| `validation_errors` | list | Schema mismatch details |

### Helpers
- `success(value, tool=..., max_bytes=..., redact_keys=..., schema=..., **meta)` — wrap success
- `failure(error, tool=..., retryable=False, retry_after=0, **meta)` — wrap failure
- `image_result(b64, mime="image/png", caption="")` — multi-modal
- `text_result(text, tool=...)` — plain text
- `multi_result(text=None, image_b64=None, data=None, ...)` — bundle
- `is_success(r)` / `is_failure(r)` — type guards
- `r.summary()` — short, prompt-safe string
- `r.to_dict()` — JSON-serializable

## Integration

```python
# tiny-retry-plus interop
from tiny_retry_plus import retry
from tiny_tool_result import failure, success

@retry(on=failure, max_attempts=3)
def flaky_call(): ...

# tiny-trace correlation
from tiny_trace_otel import get_tracer
tracer = get_tracer()
with tracer.start_span("tool.call") as span:
    r = success(value, trace_id=span.context.trace_id)
```

## Why "tiny"?

Part of the OpenClaw zero-dep, single-file ecosystem: `tiny-llm`, `tiny-llm-router`, `tiny-semantic-cache`, `tiny-prompt-cache`, `tiny-cost`, `tiny-trace`, `tiny-trace-otel`, `tiny-event-bus`, `tiny-router`, `tiny-config`, `tiny-log`, `tiny-validator`, `tiny-cli`, `tiny-rate-limiter`, `tiny-circuit-breaker`, `tiny-retry-plus`, `tiny-bounty`, `tiny-workflow-engine`, `tiny-scheduler`, and 50+ more.

## License

MIT — copy-paste, modify, ship.
