
import pytest
from ivi_water.security_utils import redact_sensitive_data

def test_redact_simple_dict():
    data = {"name": "John", "api_key": "secret123"}
    redacted = redact_sensitive_data(data)
    assert redacted["name"] == "John"
    assert redacted["api_key"] == "***REDACTED***"

def test_redact_nested_dict():
    data = {
        "user": {
            "name": "Jane",
            "auth_token": "token456"
        },
        "meta": "data"
    }
    redacted = redact_sensitive_data(data)
    assert redacted["user"]["name"] == "Jane"
    assert redacted["user"]["auth_token"] == "***REDACTED***"
    assert redacted["meta"] == "data"

def test_redact_list_of_dicts():
    data = [
        {"id": 1, "password": "pass"},
        {"id": 2, "secret": "hidden"}
    ]
    redacted = redact_sensitive_data(data)
    assert redacted[0]["id"] == 1
    assert redacted[0]["password"] == "***REDACTED***"
    assert redacted[1]["id"] == 2
    assert redacted[1]["secret"] == "***REDACTED***"

def test_redact_suffixes():
    data = {
        "my_api_key": "1",
        "user-password": "2",
        "client_secret": "3",
        "not_sensitive_key": "4" # 'key' is sensitive, 'sensitive_key' matches 'key' suffix?
    }
    # "key" is in SENSITIVE_KEYS.
    # "not_sensitive_key" ends with "_key", so it matches.
    # Wait, existing implementation matches: key_lower.endswith(f"_{sensitive}")
    # sensitive="key". "_key". So "not_sensitive_key" should be redacted.

    redacted = redact_sensitive_data(data)
    assert redacted["my_api_key"] == "***REDACTED***"
    assert redacted["user-password"] == "***REDACTED***"
    assert redacted["client_secret"] == "***REDACTED***"
    assert redacted["not_sensitive_key"] == "***REDACTED***"

def test_redact_case_insensitive():
    data = {"API_KEY": "secret"}
    redacted = redact_sensitive_data(data)
    assert redacted["API_KEY"] == "***REDACTED***"

def test_redact_recursion_limit():
    # Create circular reference
    data = {}
    data["self"] = data

    # It should hit recursion limit and return specific string
    redacted = redact_sensitive_data(data, max_depth=5)
    # Check deeply nested
    curr = redacted
    for _ in range(5):
        if isinstance(curr, dict) and "self" in curr:
            curr = curr["self"]
        else:
            break

    # The leaf should be the error message or it might have stopped earlier
    # Actually, redact_sensitive_data returns "***RECURSION LIMIT EXCEEDED***" when depth > max_depth
    # But for a recursive dict, it returns a new dict at each level until max_depth

    def check_depth(d, level):
        if level > 5:
            return d == "***RECURSION LIMIT EXCEEDED***"
        if isinstance(d, dict) and "self" in d:
            return check_depth(d["self"], level + 1)
        return False

    assert check_depth(redacted, 0)

def test_redact_non_string_keys():
    data = {1: "value", "key": "secret"}
    redacted = redact_sensitive_data(data)
    assert redacted[1] == "value"
    assert redacted["key"] == "***REDACTED***"

def test_redact_suffix_boundary():
    # "public_key" is sensitive.
    # "republic_key" -> ends with "_key" (sensitive "key").
    # What about "monkey"? ends with "key". But no "_".

    data = {
        "monkey": "banana",
        "the_key": "secret"
    }
    redacted = redact_sensitive_data(data)
    assert redacted["monkey"] == "banana"
    assert redacted["the_key"] == "***REDACTED***"

def test_redact_hyphen_match():
    data = {"x-api-key": "secret", "my-token": "secret"}
    redacted = redact_sensitive_data(data)
    assert redacted["x-api-key"] == "***REDACTED***"
    assert redacted["my-token"] == "***REDACTED***"
