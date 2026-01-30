import pytest
from ivi_water.security_utils import redact_text_content, SENSITIVE_KEYS


def test_redact_text_content_empty():
    assert redact_text_content("") == ""
    assert redact_text_content(None) is None


def test_redact_text_content_no_sensitive_data():
    text = "This is a normal log message with user_id=123"
    assert redact_text_content(text) == text


def test_redact_text_content_key_value_pairs():
    # Test standard format key=value
    text = "Connecting with api_key=secret123 and user=test"
    expected = "Connecting with api_key=***REDACTED*** and user=test"
    assert redact_text_content(text) == expected


def test_redact_text_content_colon_separator():
    # Test format key: value
    # Should preserve the colon separator
    text = "Authorization: Bearer_xyz789token"
    expected = "Authorization: ***REDACTED***"
    assert redact_text_content(text) == expected


def test_redact_text_content_json_like():
    # Should preserve the colon separator and JSON structure
    text = '{"api_key": "secret_value", "public": "data"}'
    expected = '{"api_key": "***REDACTED***", "public": "data"}'
    assert redact_text_content(text) == expected


def test_redact_text_content_multiple_matches():
    text = "api_key=abc password=123"
    expected = "api_key=***REDACTED*** password=***REDACTED***"
    assert redact_text_content(text) == expected


def test_redact_text_content_case_insensitive():
    text = "API_KEY=Secret"
    expected = "API_KEY=***REDACTED***"
    assert redact_text_content(text) == expected


def test_redact_text_content_with_quotes():
    text = "token='my-token-val'"
    expected = "token='***REDACTED***'"
    assert redact_text_content(text) == expected

    text = 'token="my-token-val"'
    expected = 'token="***REDACTED***"'
    assert redact_text_content(text) == expected


def test_authorization_header_redaction():
    # Test space-separated Bearer token (previously vulnerable)
    header = "Authorization: Bearer secret_token_12345"
    redacted = redact_text_content(header)
    assert "secret_token_12345" not in redacted
    assert "***REDACTED***" in redacted
    assert "Bearer" in redacted  # The scheme should be preserved for debugging


def test_key_value_with_spaces_colon():
    text = "api_key: value with spaces"
    redacted = redact_text_content(text)
    assert "value with spaces" not in redacted
    assert "***REDACTED***" in redacted


def test_key_value_equals_still_strict():
    # Ensure we didn't break strictly space-separated values for =
    # Use public_id as non-sensitive key to ensure it's not redacted
    text = "api_key=secret public_id=value"
    redacted = redact_text_content(text)
    assert "secret" not in redacted
    assert "public_id" in redacted # Should not consume next key
    assert "value" in redacted # public_id is not sensitive


def test_json_like_string_unquoted():
    # {key: value, ...}
    text = "{api_key: secret value, other: value}"
    redacted = redact_text_content(text)
    assert "secret value" not in redacted
    assert "other" in redacted


def test_suffix_key_preservation():
    # keys matching as suffix of other words should not trigger redaction
    # 'key' is sensitive, but 'monkey' should be preserved
    text = 'monkey="banana"'
    redacted = redact_text_content(text)
    assert 'monkey="banana"' in redacted
    assert "***REDACTED***" not in redacted

    # 'token' is sensitive, but 'public_token' (if not in sensitive list) should ideally be preserved
    # 'public_token' ends with 'token'. '_' is a word char, so \b does not match.
    text = 'public_token="safe"'
    redacted = redact_text_content(text)
    assert 'public_token="safe"' in redacted
