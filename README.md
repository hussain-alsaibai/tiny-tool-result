# tiny-tool-result

> Zero-dependency structured tool results for AI agents.

Standardized success/failure envelopes with timing, stdout/stderr capture, and metadata. Every tool in your agent loop returns the same shape — so the LLM always knows what to reason about.

```python
from tiny_tool_result import ToolResult, wrap

# Automatically wrap any function
@wrap
def fetch_data(endpoint):
    if not endpoint.startswith("http"):
        raise ValueError("invalid URL")
    return {"rows": 42}

result = fetch_data("https://api.example.com")
if result:
    print(result.output)      # {'rows': 42}
print(result.duration_ms)     # 0.12
```

## Why

AI agents fail. That's not the problem — the problem is **failing unpredictably**. When tools raise exceptions, the agent loop crashes or the LLM gets confused by inconsistent error shapes.

`tiny-tool-result` standardizes every tool call into a single envelope:

- **`success: bool`** — did it work?
- **`output: Any`** — the payload, if any
- **`error: str | None`** — human-readable error
- **`stdout` / `stderr`** — captured streams
- **`duration_ms`** — timing for observability
- **`metadata`** — arbitrary context (session, agent, retry count, ...)

## Features

- **Fluent API** — `ToolResult.ok(...)`, `ToolResult.fail(...)`, `T.ok()` oneliner
- **`@wrap` decorator** — turn any function into an envelope-returning tool
- **`from_exception`** — capture exception class + message
- **`expect()`** — return output or raise a clear ValueError
- **JSON serialization** — `to_dict()` / `to_json()` for logging and cross-process handoff
- **Zero dependencies** — stdlib only

## API

| Method | Description |
|--------|-------------|
| `ToolResult.ok(output=None, **kw)` | Success envelope |
| `ToolResult.fail(error, output=None, **kw)` | Failure envelope |
| `ToolResult.from_exception(exc)` | Build from an exception |
| `result.expect(msg=None)` | Return output, raise on failure |
| `result.to_dict()` / `result.to_json()` | Serialize for logs/handoff |
| `@wrap` | Decorator — auto-envelope any function |

## Agent Loop Pattern

```python
tools = {
    "search": wrap(search_api),
    "read": wrap(read_file),
    "write": wrap(write_file),
    "exec": wrap(run_command),
}

for step in agent_loop:
    tool_name, args = step.parse_decision()
    result = tools[tool_name](**args)
    
    if not result.success:
        # Retry with backoff, or feed error back to LLM
        prompt = f"Tool failed: {result.error}. Retry or pick another tool."
    else:
        prompt = f"Tool returned: {result.to_json()}"
```

## Zero Dependencies, Max Utility

`tiny-tool-result` is a single file, ~100 LOC, MIT licensed. Import it, wrap your tools, and stop defending against unexpected exceptions.

## License

MIT — use it, ship it, forget it.
