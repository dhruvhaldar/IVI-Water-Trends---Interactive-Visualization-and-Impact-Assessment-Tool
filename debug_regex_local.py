
import re

SENSITIVE_KEYS = {
    'api_key', 'apikey', 'key',
    'token', 'access_token', 'refresh_token', 'auth_token',
    'secret', 'client_secret',
    'password', 'passwd', 'pwd',
    'authorization', 'auth',
    'private_key', 'public_key'
}

keys_pattern = '|'.join(re.escape(k) for k in SENSITIVE_KEYS)

def redact_text_content(text: str) -> str:
    print(f"DEBUG: Input: {text}")

    # 1. Double quotes
    pattern_double = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)(")((?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    text = pattern_double.sub(r'\1\2\1\3"***REDACTED***"', text)
    print(f"DEBUG: After double: {text}")

    # 2. Single quotes
    pattern_single = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)(\')((?:[^\'\\]|\\.)*)\'',
        re.DOTALL
    )
    # NOTE: Using raw string with \' inside
    text = pattern_single.sub(r'\1\2\1\3\'***REDACTED***\'', text)
    print(f"DEBUG: After single: {text}")

    # 3. Unquoted
    pattern_unquoted = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)([^"\'\s,;}\]]+)',
        re.DOTALL
    )
    text = pattern_unquoted.sub(r'\1\2\1\3***REDACTED***', text)
    print(f"DEBUG: After unquoted: {text}")

    return text

# Test case
text = "{'password': 'pass\"word'}"
redact_text_content(text)
