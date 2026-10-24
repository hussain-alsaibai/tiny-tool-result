"""
Tests for tiny_tool_result.

38 tests covering envelope construction, redactors, trunactors, schema
validation, multi-modal helpers, and redaction/secret scrubbing in failures.
"""
import json
import pytest

from tiny_tool_result import (
    ToolResult,
    success,
    failure,
    image_result,
    text_result,
    multi_result,
    is_success,
    is_failure,
    approx_tokens,
    redact,
    validate_schema,
    truncate_value,
)


# approx_tokens

def test_approx_tokens_none():
    assert approx_tokens(None) == 0

def test_approx_tokens_short_string():
    assert approx_tokens("hi") == 1

def test_approx_tokens_long_string():
    assert approx_tokens("a" * 400) == 100

def test_approx_tokens_dict():
    # Each 80-char value is 80//4 = 20 tokens, and each 1-char key
    # contributes 1 token, so 20 + 20 + 1 + 1 = 42.
    assert approx_tokens({"a": "b" * 80, "c": "d" * 80}) == 42

def test_approx_tokens_list():
    assert approx_tokens(["ab", "cd", "ef"]) == 3


# redact

def test_redact_simple_key():
    val = {"Authorization": "Bearer xyz", "user": "ada"}
    out, hits = redact(val, ["Authorization"])
    assert out["Authorization"] == "[REDACTED]"
    assert out["user"] == "ada"
    assert "Authorization" in hits

def test_redact_glob_nested():
    val = {"data": {"password": "secret", "name": "x"}}
    out, hits = redact(val, ["*.password"])
    assert out["data"]["password"] == "[REDACTED]"
    assert "data.password" in hits

def test_redact_case_insensitive():
    val = {"API_KEY": "abc"}
    out, hits = redact(val, ["api_key"])
    assert out["API_KEY"] == "[REDACTED]"

def test_redact_list_path():
    val = {"users": [{"token": "abc"}, {"name": "ok"}]}
    out, hits = redact(val, ["*.token"])
    assert out["users"][0]["token"] == "[REDACTED]"
    assert out["users"][1]["name"] == "ok"

def test_redact_no_match():
    val = {"a": 1, "b": 2}
    out, hits = redact(val, ["missing"])
    assert out == val
    assert hits == []


# truncate_value

def test_truncate_string_under_limit():
    val, trunc, orig = truncate_value("hello", max_bytes=100)
    assert val == "hello"
    assert trunc is False
    assert orig == 5

def test_truncate_string_over_limit():
    val, trunc, orig = truncate_value("a" * 5000, max_bytes=100)
    assert trunc is True
    assert orig == 5000
    assert "TRUNCATED" in val

def test_truncate_dict_keeps_prefix():
    big = {f"k{i}": "v" * 100 for i in range(50)}
    val, trunc, orig = truncate_value(big, max_bytes=200)
    assert trunc is True
    assert isinstance(val, dict)
    assert "_truncated" in val

def test_truncate_list_halves():
    val, trunc, orig = truncate_value(list(range(1000)), max_bytes=10)
    assert trunc is True
    assert len(val) < 1000

def test_truncate_none():
    val, trunc, orig = truncate_value(None, max_bytes=10)
    assert val is None
    assert trunc is False


# validate_schema (subset)

def test_schema_type_match():
    assert validate_schema({"a": 1}, {"type": "object"}) == []

def test_schema_type_mismatch():
    errs = validate_schema("hi", {"type": "object"})
    assert len(errs) == 1
    assert errs[0].path == "$"

def test_schema_required_missing():
    errs = validate_schema({"a": 1}, {"type": "object", "required": ["b"]})
    assert any("b" in e.path for e in errs)

def test_schema_required_present():
    errs = validate_schema({"a": 1, "b": 2}, {"type": "object", "required": ["b"]})
    assert errs == []

def test_schema_nested_properties():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"age": {"type": "integer", "minimum": 0}},
            }
        },
    }
    assert validate_schema({"user": {"age": 5}}, schema) == []
    errs = validate_schema({"user": {"age": -1}}, schema)
    assert any("minimum" in e.expected for e in errs)

def test_schema_enum():
    assert validate_schema("ok", {"enum": ["ok", "err"]}) == []
    errs = validate_schema("meh", {"enum": ["ok", "err"]})
    assert errs

def test_schema_string_pattern():
    errs = validate_schema("abc-123", {"type": "string", "pattern": r"^[a-z]+-\d+$"})
    assert errs == []
    # A value that does not match the pattern is reported.
    bad = validate_schema("ABC-123", {"type": "string", "pattern": r"^[a-z]+-\d+$"})
    assert bad
    errs = validate_schema("ABC", {"type": "string", "pattern": r"^[a-z]+$"})
    assert errs

def test_schema_array_items():
    errs = validate_schema([1, 2, "x"], {"type": "array", "items": {"type": "integer"}})
    assert any("[2]" in e.path for e in errs)


# success / failure

def test_success_basic():
    r = success({"a": 1}, tool="x")
    assert r.ok is True
    assert r.value == {"a": 1}
    assert r.tool == "x"
    assert r.kind == "json"

def test_success_with_redaction():
    r = success({"Authorization": "Bearer x"}, redact_keys=["Authorization"])
    assert r.value["Authorization"] == "[REDACTED]"
    assert "Authorization" in r.redacted

def test_success_with_truncation():
    r = success({"x": "y" * 10000}, max_bytes=200)
    assert r.truncated is True
    assert r.original_bytes > 200

def test_success_with_schema():
    r = success({"id": 1}, schema={"type": "object", "required": ["id"]})
    assert r.valid
    r2 = success({"x": 1}, schema={"type": "object", "required": ["id"]})
    assert not r2.valid
    assert r2.validation_errors

def test_success_tokens_estimated():
    r = success("hello world this is a longer string")
    assert r.tokens and r.tokens > 0

def test_failure_basic():
    r = failure("oops", tool="x", retryable=True, retry_after=1.5)
    assert r.ok is False
    assert r.error == "oops"
    assert r.retryable is True
    assert r.retry_after == 1.5

def test_failure_redacts_secrets():
    r = failure("Authorization: Bearer abc123 token=xyz789 failed")
    assert "abc123" not in r.error
    assert "xyz789" not in r.error
    assert "[REDACTED]" in r.error


# Secret scrubbing regression tests
#
# A naive "key then one word" scrubber leaks the credential itself when
# the value carries an auth scheme: "Authorization: Bearer abc123" would
# redact the word "Bearer" and leave "abc123" in the message. These tests
# pin the behaviour so the leak cannot come back.

@pytest.mark.parametrize(
    "text,secret",
    [
        ("Authorization: Bearer abc123", "abc123"),
        ("Authorization: Bearer sk-abc123def456ghi789jkl", "sk-abc123def456ghi789jkl"),
        ("token=xyz789", "xyz789"),
        ("api_key: AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        ("password: hunter2", "hunter2"),
        ("client_secret: s3cr3t-value", "s3cr3t-value"),
    ],
)
def test_failure_does_not_leak_secret_values(text, secret):
    r = failure(text)
    assert secret not in r.error
    assert "[REDACTED]" in r.error


def test_failure_redacts_multiple_secrets():
    r = failure("Authorization: Bearer abc123 token=xyz789 failed")
    assert "abc123" not in r.error
    assert "xyz789" not in r.error
    assert "[REDACTED]" in r.error


def test_failure_keeps_non_secret_text():
    r = failure("connection refused after 3 retries")
    assert "connection refused" in r.error
    assert "[REDACTED]" not in r.error


# Helpers

def test_text_result():
    r = text_result("hi", tool="t")
    assert r.kind == "text"
    assert r.value == "hi"

def test_image_result():
    r = image_result("BASE64", caption="cat", tool="t")
    assert r.kind == "image"
    assert r.mime == "image/png"
    assert r.value["caption"] == "cat"

def test_multi_result_text_only():
    r = multi_result(text="hello", tool="t")
    assert r.kind == "multi"
    assert r.value == {"text": "hello"}

def test_multi_result_text_and_image():
    r = multi_result(text="see below", image_b64="B64", data={"x": 1})
    assert "text" in r.value
    assert "image_b64" in r.value
    assert r.value["data"] == {"x": 1}


# Type guards

def test_is_success_true():
    assert is_success(success(1)) is True

def test_is_success_false_on_failure():
    assert is_success(failure("x")) is False

def test_is_failure_true():
    assert is_failure(failure("x")) is True

def test_is_failure_on_non_result():
    assert is_failure({"ok": True}) is False
    assert is_success({"ok": True}) is False


# Serialization & summary

def test_to_dict_round_trip():
    r = success({"a": 1}, tool="x", latency_ms=5.0)
    d = r.to_dict()
    assert d["ok"] is True
    assert d["value"] == {"a": 1}
    assert d["tool"] == "x"
    assert d["latency_ms"] == 5.0
    json.dumps(d)

def test_summary_truncates_long_strings():
    long = "x" * 2000
    r = success(long)
    s = r.summary(max_chars=200)
    assert len(s) < 250
    assert "+" in s

def test_summary_failure():
    r = failure("bad", tool="x")
    assert "FAIL" in r.summary()
    assert "x" in r.summary()


# Pipeline order

def test_pipeline_redact_before_truncate():
    huge_secret = "a" * 10000
    r = success(
        {"password": huge_secret, "name": "ok"},
        redact_keys=["password"],
        max_bytes=100,
    )
    assert r.value["password"] == "[REDACTED]"
    assert "name" in str(r.value)


# Trace metadata passthrough

def test_trace_id_propagates():
    r = success(42, tool="x", trace_id="abc123", span_id="def456")
    assert r.trace_id == "abc123"
    assert r.span_id == "def456"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
