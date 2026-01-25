import pytest
from ivi_water.security_utils import redact_text_content

def test_redact_text_content_truncated_json_double_quote():
    # Vulnerability: Truncated JSON log usually exposes the secret because regex expects closing quote
    text = '{"password": "MySecretPassword123'
    # We expect redaction even if truncated
    # Note: Our fix will likely add a closing quote or just redact the content
    # Let's assert that the secret is NOT present
    result = redact_text_content(text)
    assert "MySecretPassword123" not in result
    assert "***REDACTED***" in result

def test_redact_text_content_truncated_json_single_quote():
    text = "{'api_key': 'secret-key-value"
    result = redact_text_content(text)
    assert "secret-key-value" not in result
    assert "***REDACTED***" in result

def test_redact_text_content_truncated_escaped_quote():
    # Case where truncation happens after an escaped quote
    # Input: {"password": "My \"Secret\" Pass
    text = r'{"password": "My \"Secret\" Pass'
    result = redact_text_content(text)
    assert "Secret" not in result
    assert "Pass" not in result
    assert "***REDACTED***" in result

def test_redact_text_content_complete_still_works():
    # Regression check within this file
    text = '{"password": "MySecretPassword123"}'
    result = redact_text_content(text)
    assert "MySecretPassword123" not in result
    assert '{"password": "***REDACTED***"}' in result
