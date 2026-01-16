"""
Security Utilities Module

This module provides helper functions for security-related tasks,
such as redacting sensitive information from logs.
"""

import re
import hmac
import hashlib
from typing import Dict, Any, Union, List, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# List of keys that are considered sensitive and should be redacted
SENSITIVE_KEYS = {
    'api_key', 'apikey', 'key',
    'token', 'access_token', 'refresh_token', 'auth_token',
    'secret', 'client_secret',
    'password', 'passwd', 'pwd',
    'authorization', 'auth',
    'private_key', 'public_key',
    # Added hyphenated variants for broader coverage
    'api-key', 'access-token', 'refresh-token', 'auth-token',
    'client-secret', 'x-api-key', 'x-access-token', 'x-auth-token'
}

# Regex for validating safe identifiers (alphanumeric, -, _)
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Maximum length for identifiers to prevent DoS/memory issues
MAX_ID_LENGTH = 128

def redact_sensitive_data(data: Any, max_depth: int = 50, _current_depth: int = 0) -> Any:
    """
    Recursively redact sensitive keys in a dictionary or list.

    Args:
        data: The input data (dict, list, or other types).
        max_depth: Maximum recursion depth to prevent StackOverflow (default: 50).
        _current_depth: Internal counter for recursion depth.

    Returns:
        The data with sensitive values replaced by '***REDACTED***'.
        If recursion limit is reached, returns '***RECURSION LIMIT EXCEEDED***'.
    """
    if _current_depth > max_depth:
        return '***RECURSION LIMIT EXCEEDED***'

    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if isinstance(key, str):
                # Check exact match or part of snake_case/kebab-case
                is_sensitive = False
                key_lower = key.lower()
                if key_lower in SENSITIVE_KEYS:
                    is_sensitive = True
                else:
                    for sensitive in SENSITIVE_KEYS:
                        # Check for _sensitive and -sensitive suffixes
                        if key_lower == sensitive or \
                           key_lower.endswith(f"_{sensitive}") or \
                           key_lower.endswith(f"-{sensitive}"):
                            is_sensitive = True
                            break

                if is_sensitive:
                    redacted[key] = '***REDACTED***'
                else:
                    redacted[key] = redact_sensitive_data(value, max_depth, _current_depth + 1)
            else:
                redacted[key] = redact_sensitive_data(value, max_depth, _current_depth + 1)
        return redacted

    elif isinstance(data, list):
        return [redact_sensitive_data(item, max_depth, _current_depth + 1) for item in data]

    return data

def redact_url(url: str) -> str:
    """
    Redact credentials from a URL.

    Args:
        url: The URL string to redact.

    Returns:
        The URL with credentials masked (e.g. user:***REDACTED***@host).
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)

        # Redact password in netloc
        if parsed.password:
             # Reconstruct netloc with redacted password
             user = parsed.username
             host = parsed.hostname
             port = parsed.port

             new_netloc = f"{user}:***REDACTED***@{host}"
             if port:
                 new_netloc += f":{port}"

             parsed = parsed._replace(netloc=new_netloc)

        # Redact sensitive query parameters
        if parsed.query:
            query_params = parse_qsl(parsed.query, keep_blank_values=True)
            redacted_params = []

            for key, value in query_params:
                is_sensitive = False
                key_lower = key.lower()

                # Check for exact match or snake_case suffix
                if key_lower in SENSITIVE_KEYS:
                    is_sensitive = True
                else:
                    for sensitive in SENSITIVE_KEYS:
                        if key_lower == sensitive or key_lower.endswith(f"_{sensitive}"):
                            is_sensitive = True
                            break

                if is_sensitive:
                    redacted_params.append((key, '***REDACTED***'))
                else:
                    redacted_params.append((key, value))

            # Allow * in value (safe='*') to prevent encoding of ***REDACTED***
            new_query = urlencode(redacted_params, doseq=True, safe='*')
            parsed = parsed._replace(query=new_query)

        return urlunparse(parsed)
    except Exception:
        # If parsing fails, return original URL (safer than returning empty or partial)
        # But for security, maybe we should return a placeholder?
        # Standard practice is to try best effort.
        return url

def validate_safe_id(identifier: str) -> str:
    """
    Validate that an identifier contains only safe characters.

    Allowed characters: Alphanumeric (a-z, A-Z, 0-9), hyphens (-), and underscores (_).
    This helps prevent injection attacks and path traversal issues.

    Args:
        identifier: The string identifier to validate.

    Returns:
        The validated identifier (stripped of whitespace).

    Raises:
        ValueError: If the identifier is empty, too long, or contains invalid characters.
    """
    if not isinstance(identifier, str):
        raise ValueError("Identifier must be a string")

    if len(identifier) > MAX_ID_LENGTH:
        raise ValueError(f"Identifier exceeds maximum length of {MAX_ID_LENGTH} characters")

    clean_id = identifier.strip()
    if not clean_id:
        raise ValueError("Identifier cannot be empty")

    if not SAFE_ID_PATTERN.match(clean_id):
        raise ValueError(
            f"Invalid identifier '{clean_id}'. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )

    return clean_id

def hash_data(data: str, key: Optional[str] = None) -> str:
    """
    Create a SHA-256 hash or HMAC-SHA-256 of the input string.

    Args:
        data: Input string to hash.
        key: Optional secret key for HMAC. If provided, HMAC-SHA256 is used.
             If None, standard SHA-256 hash is used.

    Returns:
        Hexadecimal representation of the hash/HMAC.
    """
    if key:
        # Use HMAC-SHA256 for keyed hashing (prevention of length extension attacks)
        # and stronger integrity verification
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key

        return hmac.new(key_bytes, data.encode('utf-8'), hashlib.sha256).hexdigest()

    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def redact_text_content(text: str) -> str:
    """
    Redact sensitive information from unstructured text (logs, headers, etc.).

    This function searches for patterns like 'key=value', 'key: value',
    or '"key": "value"' where the key is in the sensitive keys list.

    Args:
        text: The input text to redact.

    Returns:
        The text with sensitive values replaced by '***REDACTED***'.
    """
    if not text or not isinstance(text, str):
        return text

    # Pattern explanation:
    # (?i)          : Case-insensitive
    # \b            : Word boundary (start of key)
    # (KEY1|KEY2...) : Match any sensitive key
    # \b            : Word boundary (end of key)
    # \s*[:=]\s*    : Separator (: or =) with optional whitespace
    # (["']?)       : Capture group 2: Optional opening quote
    # (.*?)         : Capture group 3: The value (non-greedy)
    # \2            : Match the closing quote (same as group 2)
    # (?=[\s,;}]|$) : Lookahead for separator (space, comma, semicolon, closing brace) or end of string

    # We construct the regex dynamically based on SENSITIVE_KEYS
    keys_pattern = '|'.join(re.escape(k) for k in SENSITIVE_KEYS)

    # Simple pattern for standard assignments (key=value, key: value) without internal spaces/commas in value
    # We handle quoted values specially

    # 1. Match quoted values: key="value with spaces"
    pattern_quoted = re.compile(
        r'(?i)\b(' + keys_pattern + r')\b\s*[:=]\s*(["\'])(.*?)\2',
        re.DOTALL
    )

    # 2. Match unquoted values: key=value (stops at space/comma/semicolon/newline)
    # The first definition of pattern_unquoted was redundant and confusing.

    # Apply redaction
    # For quoted: Replace group 3 with ***REDACTED***
    # Use \g<0> approach to preserve separator?
    # No, we can capture the separator group if we modify regex.
    # Current regex: \b(KEY)\b\s*[:=]\s*(["'])(.*?)\2
    # It consumes the separator.
    # To fix test_json_like failure where separator changed from : to =,
    # we need to capture the separator.

    # Redefine patterns to capture separator and surrounding whitespace
    # Group 1: Key (optionally quoted)
    # Group 2: Separator with optional surrounding whitespace
    # Group 3: Opening quote
    # Group 4: Value

    # We need to handle optional quotes around the key for JSON-like strings
    # "key": "value" or 'key': 'value'

    # We use two patterns: one for double quotes, one for single quotes,
    # to correctly handle escaping within each type.

    # 1. Double quotes: "value" - handles escaped double quotes \"
    pattern_double = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)(")((?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    text = pattern_double.sub(r'\1\2\1\3"***REDACTED***"', text)

    # 2. Single quotes: 'value' - handles escaped single quotes \'
    pattern_single = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)(\')((?:[^\'\\]|\\.)*)\'',
        re.DOTALL
    )
    # Note: Use plain string with ' for replacement to avoid double escaping issues
    text = pattern_single.sub(r"\1\2\1\3'***REDACTED***'", text)

    # For unquoted: Replace group 4 (value) with ***REDACTED***
    # Group 1: Optional Quote
    # Group 2: Key
    # Group 3: Separator with optional surrounding whitespace
    # Group 4: Value (non-whitespace, non-separator chars)
    pattern_unquoted = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)([^"\'\s,;}\]]+)',
        re.DOTALL
    )

    text = pattern_unquoted.sub(r'\1\2\1\3***REDACTED***', text)

    return text
