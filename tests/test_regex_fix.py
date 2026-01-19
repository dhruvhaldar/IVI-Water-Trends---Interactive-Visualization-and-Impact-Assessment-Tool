import pytest
from ivi_water.security_utils import redact_text_content


class TestRegexFix:
    """Test the fix for regex-based redaction of JSON-like strings."""

    def test_redact_simple_secret(self):
        text = '{"password": "secret123"}'
        expected = '{"password": "***REDACTED***"}'
        assert redact_text_content(text) == expected

    def test_redact_escaped_quote(self):
        # Input: {"password": "secret\"pass"}
        text = '{"password": "secret\\"pass"}'
        expected = '{"password": "***REDACTED***"}'
        assert redact_text_content(text) == expected

    def test_redact_multiple_escaped_quotes(self):
        # Input: {"api_key": "key_with_\"quotes\"_inside"}
        text = '{"api_key": "key_with_\\"quotes\\"_inside"}'
        expected = '{"api_key": "***REDACTED***"}'
        assert redact_text_content(text) == expected

    def test_redact_escaped_backslash(self):
        # Input: {"password": "secret\\pass"} -> secret\pass
        # In Python string: "secret\\\\pass"
        text = '{"password": "secret\\\\pass"}'
        expected = '{"password": "***REDACTED***"}'
        assert redact_text_content(text) == expected

    def test_redact_json_list(self):
        text = '[{"password": "pass1"}, {"password": "pass2"}]'
        expected = '[{"password": "***REDACTED***"}, {"password": "***REDACTED***"}]'
        assert redact_text_content(text) == expected

    def test_single_quoted_with_double_quotes(self):
        # Input: {'password': 'pass"word'}
        text = "{'password': 'pass\"word'}"
        expected = "{'password': '***REDACTED***'}"
        result = redact_text_content(text)
        print(f"FAILED RESULT: {result}")
        assert result == expected

    def test_double_quoted_with_single_quotes(self):
        # Input: {"password": "pass'word"}
        text = '{"password": "pass\'word"}'
        expected = '{"password": "***REDACTED***"}'
        assert redact_text_content(text) == expected

    def test_false_positive_prevention(self):
        # Ensure it doesn't over-redact
        text = '{"public_data": "not_secret", "password": "secret"}'
        expected = '{"public_data": "not_secret", "password": "***REDACTED***"}'
        assert redact_text_content(text) == expected
