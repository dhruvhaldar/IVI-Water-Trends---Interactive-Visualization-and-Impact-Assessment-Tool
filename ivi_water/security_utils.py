"""
Security Utilities Module

This module provides functions for sensitive data handling, including
redaction of logs and secure hashing for cache keys.
"""

import hashlib
import json
from typing import Any, Dict, List, Union

# Keys that are considered sensitive and should be redacted
SENSITIVE_KEYS = {
    'password', 'api_key', 'apikey', 'access_token', 'token',
    'secret', 'auth', 'authorization', 'credential', 'private_key',
    'client_secret', 'client_id'
}

def redact_sensitive_data(data: Any) -> Any:
    """
    Redact sensitive information from data structures for safe logging.

    recursively traverses dictionaries and lists, replacing values of
    keys that match SENSITIVE_KEYS with '[REDACTED]'.

    Args:
        data: The data to redact (dict, list, or other types)

    Returns:
        The redacted data structure (copy of original)
    """
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if isinstance(key, str) and key.lower() in SENSITIVE_KEYS:
                redacted[key] = '[REDACTED]'
            else:
                redacted[key] = redact_sensitive_data(value)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    else:
        return data

def hash_data(data: Any) -> str:
    """
    Create a secure SHA-256 hash of the input data.

    This is useful for creating cache keys or identifiers without exposing
    the underlying data values.

    Args:
        data: The data to hash (will be converted to string/bytes)

    Returns:
        Hex digest string of the SHA-256 hash
    """
    if isinstance(data, (dict, list)):
        # Sort keys for consistent hashing of dicts
        try:
            serialized = json.dumps(data, sort_keys=True)
        except (TypeError, ValueError):
            serialized = str(data)
    else:
        serialized = str(data)

    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
