from ivi_water.security_utils import redact_text_content


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


def test_redact_text_content_truncated_json():
    # Test truncated JSON where closing quote is missing
    text = '{"api_key": "super_secret_value_that_got_trun'
    # The function should close the quote in the redacted output
    expected = '{"api_key": "***REDACTED***"'
    assert redact_text_content(text) == expected


def test_redact_text_content_mismatched_quotes():
    # Test mismatched quotes (e.g. log typo or truncation)
    text = "api_key=\"super_secret_value'"
    expected = 'api_key="***REDACTED***"'
    assert redact_text_content(text) == expected


def test_redact_text_content_unquoted_leak():
    # Test value starting with quote but no closing quote (end of string)
    text = 'api_key="secret'
    expected = 'api_key="***REDACTED***"'
    assert redact_text_content(text) == expected
