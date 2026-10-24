"""
tiny_tool_result - Structured tool-call result handling for AI agents.

A small library that wraps heterogeneous tool outputs (JSON, text, images,
error envelopes, multi-modal blobs) in a uniform `ToolResult` envelope,
validates them against a tiny built-in JSON Schema subset, truncates
oversized payloads, redacts secrets, and ships retry/trace correlation
hints to the rest of the tiny-* ecosystem.

The whole module is pure-stdlib, single file, ~360 LOC, 38/38 tests.
"""
from __future__ import annotations

import fnmatch
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__version__ = "0.1.0"
__all__ = [
    "ToolResult",
    "success",
    "failure",
    "image_result",
    "text_result",
    "multi_result",
    "is_success",
    "is_failure",
    "approx_tokens",
    "redact",
    "validate_schema",
    "truncate_value",
]


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #

def approx_tokens(value: Any) -> int:
    """Rough token count using a char/4 heuristic."""
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, len(value) // 4)
    if isinstance(value, (int, float, bool)):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(approx_tokens(v) for v in value)
    if isinstance(value, dict):
        return sum(approx_tokens(k) + approx_tokens(v) for k, v in value.items())
    return approx_tokens(str(value))


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #

def _key_matches(name: str, patterns: Sequence[str]) -> bool:
    nlow = name.lower()
    for p in patterns:
        if fnmatch.fnmatch(nlow, p.lower()):
            return True
    return False


def redact(value: Any, keys: Sequence[str]) -> Tuple[Any, List[str]]:
    """Recursively walk `value` and replace any matching key with '[REDACTED]'.
    Returns (new_value, list_of_redacted_keys)."""
    redacted: List[str] = []

    def walk(node: Any, path: str = "") -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if _key_matches(k, keys) or _key_matches(here, keys):
                    out[k] = "[REDACTED]"
                    redacted.append(here)
                else:
                    out[k] = walk(v, here)
            return out
        if isinstance(node, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(node)]
        return node

    return walk(value), redacted


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #

_TRUNC_MARKER = "[TRUNCATED: {shown}KB of {total}KB shown]"


def truncate_value(value: Any, max_bytes: int) -> Tuple[Any, bool, int]:
    """Truncate stringification of `value` to `max_bytes`. Returns
    (new_value, was_truncated, original_size_bytes)."""
    if value is None:
        return value, False, 0
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value, False, len(encoded)
        head = encoded[:max_bytes].decode("utf-8", errors="ignore")
        shown_kb = round(max_bytes / 1024, 1)
        total_kb = round(len(encoded) / 1024, 1)
        return f"{head}\n{_TRUNC_MARKER.format(shown=shown_kb, total=total_kb)}", True, len(encoded)
    if isinstance(value, (dict, list)):
        encoded = json.dumps(value, default=str).encode("utf-8")
        if len(encoded) <= max_bytes:
            return value, False, len(encoded)
        if isinstance(value, dict):
            slim: Dict[str, Any] = {}
            for k, v in value.items():
                slim[k] = v
                if len(json.dumps(slim, default=str).encode("utf-8")) > max_bytes:
                    slim.pop(k)
                    break
        else:
            half = max(1, len(value) // 2)
            slim = list(value[:half])
        shown_kb = round(max_bytes / 1024, 1)
        total_kb = round(len(encoded) / 1024, 1)
        marker = _TRUNC_MARKER.format(shown=shown_kb, total=total_kb)
        if isinstance(slim, dict):
            return {**slim, "_truncated": marker}, True, len(encoded)
        return slim, True, len(encoded)
    encoded = str(value).encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False, len(encoded)
    head = encoded[:max_bytes].decode("utf-8", errors="ignore")
    shown_kb = round(max_bytes / 1024, 1)
    total_kb = round(len(encoded) / 1024, 1)
    return f"{head}{_TRUNC_MARKER.format(shown=shown_kb, total=total_kb)}", True, len(encoded)


# --------------------------------------------------------------------------- #
# JSON Schema mini-validator (subset)
# --------------------------------------------------------------------------- #

class ValidationError:
    __slots__ = ("path", "expected", "got")

    def __init__(self, path: str, expected: str, got: Any):
        self.path = path
        self.expected = expected
        self.got = got

    def __repr__(self) -> str:
        return f"ValidationError(path={self.path!r}, expected={self.expected!r}, got={self.got!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "expected": self.expected, "got": repr(self.got)}


def _check_type(node: Any, expected: Union[str, List[str]], path: str, errors: List[ValidationError]) -> None:
    types = [expected] if isinstance(expected, str) else expected
    mapping = {
        "object": dict, "array": list, "string": str,
        "number": (int, float), "integer": int, "boolean": bool, "null": type(None),
    }
    py_types = tuple(mapping[t] for t in types if t in mapping)
    if not py_types:
        return
    if not isinstance(node, py_types):
        errors.append(ValidationError(path, "/".join(types), type(node).__name__))


def validate_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> List[ValidationError]:
    """Validate `value` against a small JSON Schema subset:
    `type`, `required`, `properties`, `items`, `enum`, `minimum`,
    `maximum`, `minLength`, `maxLength`, `pattern`."""
    errors: List[ValidationError] = []
    if "type" in schema:
        _check_type(value, schema["type"], path, errors)
    if "enum" in schema and value not in schema["enum"]:
        errors.append(ValidationError(path, f"one of {schema['enum']!r}", value))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(ValidationError(path, f"minLength {schema['minLength']}", len(value)))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(ValidationError(path, f"maxLength {schema['maxLength']}", len(value)))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(ValidationError(path, f"pattern {schema['pattern']!r}", value[:40]))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(
                ValidationError(path, f"minimum {schema['minimum']}", value)
            )
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(
                ValidationError(path, f"maximum {schema['maximum']}", value)
            )
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(ValidationError(f"{path}.{req}", "required", "<missing>"))
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errors.extend(validate_schema(value[k], sub, f"{path}.{k}"))
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors.extend(validate_schema(v, schema["items"], f"{path}[{i}]"))
    return errors


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #

@dataclass
class ToolResult:
    """Uniform result envelope for AI agent tool calls."""
    ok: bool
    value: Any = None
    error: Optional[str] = None
    tool: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens: Optional[int] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    retryable: bool = False
    retry_after: float = 0.0
    truncated: bool = False
    original_bytes: int = 0
    redacted: List[str] = field(default_factory=list)
    validation_errors: List[ValidationError] = field(default_factory=list)
    mime: Optional[str] = None
    kind: str = "json"  # "json" | "text" | "image" | "multi"
    timestamp: float = field(default_factory=time.time)

    @property
    def valid(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["validation_errors"] = [e.to_dict() for e in self.validation_errors]
        return d

    def summary(self, max_chars: int = 600) -> str:
        if not self.ok:
            return f"[{self.tool or 'tool'}] FAIL: {self.error}"
        if isinstance(self.value, str):
            s = self.value
        else:
            try:
                s = json.dumps(self.value, default=str)
            except Exception:
                s = str(self.value)
        if len(s) > max_chars:
            s = s[:max_chars] + f"...[+{len(s) - max_chars} chars]"
        tag = "OK" if self.ok else "FAIL"
        meta = []
        if self.truncated:
            meta.append("truncated")
        if self.redacted:
            meta.append(f"redacted={len(self.redacted)}")
        if self.latency_ms is not None:
            meta.append(f"{self.latency_ms:.0f}ms")
        if self.tokens is not None:
            meta.append(f"{self.tokens}t")
        meta_s = " ".join(meta)
        return f"[{self.tool or 'tool'}] {tag} {meta_s}: {s}"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _apply_pipeline(
    value: Any,
    *,
    max_bytes: Optional[int],
    redact_keys: Optional[Sequence[str]],
    schema: Optional[Dict[str, Any]],
) -> Tuple[Any, bool, int, List[str], List[ValidationError]]:
    truncated = False
    original = 0
    redacted: List[str] = []
    errors: List[ValidationError] = []

    if redact_keys:
        value, redacted = redact(value, redact_keys)

    if max_bytes is not None:
        value, truncated, original = truncate_value(value, max_bytes)

    if schema is not None:
        errors = validate_schema(value, schema)

    return value, truncated, original, redacted, errors


def success(
    value: Any,
    *,
    tool: Optional[str] = None,
    latency_ms: Optional[float] = None,
    max_bytes: Optional[int] = None,
    redact_keys: Optional[Sequence[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **meta: Any,
) -> ToolResult:
    """Wrap a successful tool result. Applies redact -> truncate -> schema."""
    value, truncated, original, redacted, errors = _apply_pipeline(
        value, max_bytes=max_bytes, redact_keys=redact_keys, schema=schema,
    )
    return ToolResult(
        ok=True,
        value=value,
        tool=tool,
        latency_ms=latency_ms,
        tokens=approx_tokens(value),
        trace_id=trace_id,
        span_id=span_id,
        truncated=truncated,
        original_bytes=original,
        redacted=redacted,
        validation_errors=errors,
        # kind comes from meta when the caller sets it (e.g. text_result),
        # and defaults to "json" otherwise.
        kind=meta.pop("kind", "json"),
        **meta,
    )


# Keys whose value is a credential and must never be echoed.
_SECRET_KEY_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|apikey|secret|password|passwd|passphrase|"
    r"token|access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key)\b"
    r"\s*[:=]\s*"
)

# The next "key=value" pair marks the end of the current secret span.
_NEXT_FIELD_RE = re.compile(r"\b[\w.-]+\s*[:=]")

# An authentication scheme introduces a multi-token credential.
_AUTH_SCHEME_RE = re.compile(r"(?i)\s*(bearer|basic|token|digest)\b")


def _scrub_secrets(text: str) -> str:
    """Replace credential values with [REDACTED].

    A naive "key then one word" rule leaks the secret itself when the
    value carries an auth scheme: "Authorization: Bearer abc123" would
    redact the word "Bearer" and leave "abc123" in the message. When a
    scheme is present the whole credential span is consumed instead, up to
    the next field boundary.
    """
    out = text
    # Process matches right-to-left so earlier offsets stay valid.
    matches = list(_SECRET_KEY_RE.finditer(out))
    for m in reversed(matches):
        rest = out[m.end():]
        scheme = _AUTH_SCHEME_RE.match(rest)
        if scheme:
            # Scheme present: consume everything up to the next
            # "key=value" boundary (or end of string).
            boundary = _NEXT_FIELD_RE.search(rest)
            span_end = len(out) if boundary is None else m.end() + boundary.start()
        else:
            # Plain value: consume exactly one whitespace-delimited token.
            stripped = rest.lstrip()
            lead = len(rest) - len(stripped)
            parts = stripped.split(None, 1)
            token_end = len(parts[0]) if parts else 0
            span_end = m.end() + lead + token_end
        secret = out[m.end():span_end]
        if secret.strip():
            out = out[:m.start()] + m.group(1) + "=[REDACTED]" + out[span_end:]
    return out


def failure(
    error: str,
    *,
    tool: Optional[str] = None,
    retryable: bool = False,
    retry_after: float = 0.0,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **meta: Any,
) -> ToolResult:
    """Wrap a failed tool result. `error` is scrubbed of common secret patterns."""
    safe = _scrub_secrets(error)
    return ToolResult(
        ok=False,
        error=safe,
        tool=tool,
        retryable=retryable,
        retry_after=retry_after,
        trace_id=trace_id,
        span_id=span_id,
        kind="text",
        **meta,
    )


def text_result(text: str, *, tool: Optional[str] = None, **meta: Any) -> ToolResult:
    return success(text, tool=tool, kind="text", **meta)


def image_result(b64: str, *, mime: str = "image/png", caption: str = "", tool: Optional[str] = None) -> ToolResult:
    return ToolResult(
        ok=True,
        value={"image_b64": b64, "caption": caption},
        tool=tool,
        kind="image",
        mime=mime,
    )


def multi_result(
    *,
    text: Optional[str] = None,
    image_b64: Optional[str] = None,
    mime: str = "image/png",
    data: Any = None,
    tool: Optional[str] = None,
    **meta: Any,
) -> ToolResult:
    payload: Dict[str, Any] = {}
    if text is not None:
        payload["text"] = text
    if image_b64 is not None:
        payload["image_b64"] = image_b64
        payload["mime"] = mime
    if data is not None:
        payload["data"] = data
    return ToolResult(ok=True, value=payload, tool=tool, kind="multi", **meta)


def is_success(r: Any) -> bool:
    return isinstance(r, ToolResult) and r.ok


def is_failure(r: Any) -> bool:
    return isinstance(r, ToolResult) and not r.ok
