import pytest
from ivi_water.security_utils import redact_text_content

def test_redact_suffix_key_with_underscore():
    # 'token' is sensitive. 'my_token' should be redacted because '_' is a separator.
    text = 'my_token="secret"'
    expected = 'my_token="***REDACTED***"'
    assert redact_text_content(text) == expected

def test_redact_suffix_key_with_hyphen():
    # 'token' is sensitive. 'my-token' should be redacted because '-' is a separator.
    text = 'my-token="secret"'
    expected = 'my-token="***REDACTED***"'
    assert redact_text_content(text) == expected

def test_redact_suffix_key_with_dot():
    # 'token' is sensitive. 'my.token' should be redacted because '.' is a separator.
    text = 'my.token="secret"'
    expected = 'my.token="***REDACTED***"'
    assert redact_text_content(text) == expected

def test_redact_suffix_key_with_space():
    # 'token' is sensitive. ' token' should be redacted because ' ' is a separator.
    text = ' token="secret"'
    expected = ' token="***REDACTED***"'
    assert redact_text_content(text) == expected

def test_redact_suffix_key_with_quote():
    # 'token' is sensitive. '"token"' should be redacted because '"' is a separator.
    text = '"token"="secret"'
    expected = '"token"="***REDACTED***"'
    assert redact_text_content(text) == expected

def test_no_redaction_for_letter_suffix():
    # 'token' is sensitive. 'monkey' (ends with 'key') should NOT be redacted.
    # 'key' is sensitive.
    # 'monkey="banana"'
    text = 'monkey="banana"'
    expected = 'monkey="banana"'
    assert redact_text_content(text) == expected

def test_no_redaction_for_number_suffix():
    # 'key' is sensitive. '1key' should NOT be redacted if we treat numbers as part of word.
    text = '1key="val"'
    expected = '1key="val"'
    assert redact_text_content(text) == expected
