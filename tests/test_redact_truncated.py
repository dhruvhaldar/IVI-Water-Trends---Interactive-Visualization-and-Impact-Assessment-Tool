import pytest
from ivi_water.security_utils import redact_text_content

def test_redact_truncated_json_double_quote():
    # Input: {"api_key": "super_secret... (truncated)
    text = '{"api_key": "super_secret_value_that_is_truncated'
    # Expected: {"api_key": "***REDACTED***" (we add the closing quote effectively)
    # Or at least we expect the secret to be gone.
    redacted = redact_text_content(text)
    assert "super_secret" not in redacted
    assert "***REDACTED***" in redacted

def test_redact_truncated_json_single_quote():
    # Input: {'api_key': 'super_secret... (truncated)
    text = "{'api_key': 'super_secret_value_that_is_truncated"
    redacted = redact_text_content(text)
    assert "super_secret" not in redacted
    assert "***REDACTED***" in redacted

def test_redact_truncated_json_escaped_quote():
    # Input: {"api_key": "secret\"val... (truncated)
    text = '{"api_key": "secret\\"val'
    redacted = redact_text_content(text)
    assert "secret" not in redacted
    assert "***REDACTED***" in redacted
