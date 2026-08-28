import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

class ToolResult:
    """
    Zero-dependency structured tool results for AI agents.
    
    Standardized success/failure envelopes with timing, stdout/stderr capture,
    and metadata. Built to make agent loops predictable: every tool returns
    the same shape, so the LLM always knows what to reason about.
    
    Usage:
        result = ToolResult.success({"rows": 42})
        if not result.success:
            print(result.error)
    """
    __slots__ = ("success", "output", "error", "stdout", "stderr", "duration_ms", "metadata")

    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = duration_ms
        self.metadata = metadata or {}

    @classmethod
    def ok(cls, output: Any = None, **kwargs) -> "ToolResult":
        return cls(success=True, output=output, **kwargs)

    @classmethod
    def fail(cls, error: str, output: Any = None, **kwargs) -> "ToolResult":
        return cls(success=False, error=error, output=output, **kwargs)

    @classmethod
    def from_exception(cls, exc: Exception, **kwargs) -> "ToolResult":
        return cls(
            success=False,
            error=f"{type(exc).__name__}: {str(exc)}",
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)

    @property
    def is_ok(self) -> bool:
        return self.success

    def expect(self, msg: Optional[str] = None) -> Any:
        """Return output or raise ValueError with error message."""
        if not self.success:
            raise ValueError(msg or self.error or "Tool failed")
        return self.output

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR: {self.error}"
        return f"<ToolResult {status} ({self.duration_ms:.0f}ms)>"


class T:
    """Standalone helper for oneliners: T.ok(...), T.fail(...)"""
    ok = ToolResult.ok
    fail = ToolResult.fail


def wrap(fn=None, *, include_runtime: bool = True):
    """
    Decorator that turns any function into a ToolResult-returning tool.
    
    Examples:
        @wrap
        def divide(a, b):
            return a / b

        @wrap(include_runtime=False)
        def fast_tool(x):
            return x * 2

    On exception, returns ToolResult.fail with a str() of the error.
    """
    import time as _time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = _time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = (_time.perf_counter() - start) * 1000 if include_runtime else 0.0
                return ToolResult.ok(result, duration_ms=duration)
            except Exception as e:
                duration = (_time.perf_counter() - start) * 1000 if include_runtime else 0.0
                return ToolResult.fail(str(e), duration_ms=duration)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


if __name__ == "__main__":
    # Examples
    @wrap
    def risky(x, y):
        if y == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return x / y

    r1 = risky(10, 2)
    print(r1)          # <ToolResult OK (0.08ms)>
    print(r1.output)   # 5.0
    print(r1.to_json())

    r2 = risky(10, 0)
    print(r2)          # <ToolResult ERR: cannot divide by zero (0.01ms)>
    print(r2.success)  # False
