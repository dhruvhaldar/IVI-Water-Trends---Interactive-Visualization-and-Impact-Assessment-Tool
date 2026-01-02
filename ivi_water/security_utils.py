"""
Security Utilities Module

This module provides helper functions for security-related tasks,
such as redacting sensitive information from logs.
"""

from typing import Dict, Any, Union, List
from urllib.parse import urlparse, urlunparse

# List of keys that are considered sensitive and should be redacted
SENSITIVE_KEYS = {
    'api_key', 'apikey', 'key',
    'token', 'access_token', 'refresh_token', 'auth_token',
    'secret', 'client_secret',
    'password', 'passwd', 'pwd',
    'authorization', 'auth',
    'private_key', 'public_key'
}

def redact_sensitive_data(data: Any) -> Any:
    """
    Recursively redact sensitive keys in a dictionary or list.

    Args:
        data: The input data (dict, list, or other types).

    Returns:
        The data with sensitive values replaced by '***REDACTED***'.
    """
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if isinstance(key, str) and any(s == key.lower() or s in key.lower().split('_') for s in SENSITIVE_KEYS):
                 # Check exact match or part of snake_case (e.g. 'api_key', 'my_password')
                 # But avoid redacting things like 'location_key' if 'key' is in sensitive list?
                 # Let's use a slightly more robust check.
                 # If the key exactly matches or ends with a sensitive suffix like '_key', '_token'
                 # Or if it is in the explicit list.

                 is_sensitive = False
                 key_lower = key.lower()
                 if key_lower in SENSITIVE_KEYS:
                     is_sensitive = True
                 else:
                     for sensitive in SENSITIVE_KEYS:
                         if key_lower == sensitive or key_lower.endswith(f"_{sensitive}"):
                             is_sensitive = True
                             break

                 if is_sensitive:
                     redacted[key] = '***REDACTED***'
                 else:
                     redacted[key] = redact_sensitive_data(value)
            else:
                redacted[key] = redact_sensitive_data(value)
        return redacted

    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]

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
        if parsed.password:
             # Reconstruct netloc with redacted password
             user = parsed.username
             host = parsed.hostname
             port = parsed.port

             new_netloc = f"{user}:***REDACTED***@{host}"
             if port:
                 new_netloc += f":{port}"

             parsed = parsed._replace(netloc=new_netloc)
             return urlunparse(parsed)
        return url
    except Exception:
        # If parsing fails, return original URL (safer than returning empty or partial)
        # But for security, maybe we should return a placeholder?
        # Standard practice is to try best effort.
        return url
