"""
Security Utilities Module

This module provides functions for handling sensitive data, including
redaction for logging and secure hashing.
"""

import hashlib
import json
from typing import Any, Dict, List, Union, Optional

# Keys that are considered sensitive and should be redacted
SENSITIVE_KEYS = {
    'api_key', 'token', 'password', 'secret', 'auth', 'authorization',
    'access_token', 'refresh_token', 'client_secret', 'key'
}

def redact_sensitive_data(data: Any) -> Any:
    """
    Recursively redact sensitive keys in dictionaries and lists.

    Args:
        data: The data structure to redact (dict, list, or primitive)

    Returns:
        A copy of the data with sensitive values masked.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(key, str) and key.lower() in SENSITIVE_KEYS:
                result[key] = '***REDACTED***'
            else:
                result[key] = redact_sensitive_data(value)
        return result
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    else:
        return data

def hash_data(data: Any) -> str:
    """
    Create a consistent SHA-256 hash of the input data.

    Args:
        data: Input data (will be converted to string/json)

    Returns:
        Hex digest of the hash.
    """
    try:
        # Sort keys for consistent JSON representation
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    except (TypeError, ValueError):
        # Fallback for non-serializable data
        return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
