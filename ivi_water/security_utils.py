"""
Security Utilities Module

This module provides helper functions for security-related tasks,
such as redacting sensitive information from logs.
"""

import re
import hmac
import hashlib
import base64
from html.parser import HTMLParser
from typing import Dict, Any, Union, List, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# List of keys that are considered sensitive and should be redacted
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "key",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "authorization",
    "auth",
    "private_key",
    "public_key",
    # Added hyphenated variants for broader coverage
    "api-key",
    "access-token",
    "refresh-token",
    "auth-token",
    "client-secret",
    "x-api-key",
    "x-access-token",
    "x-auth-token",
}

# Regex for validating safe identifiers (alphanumeric, -, _)
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Maximum length for identifiers to prevent DoS/memory issues
MAX_ID_LENGTH = 128

# Standard Content Security Policy for HTML reports/dashboards
# Allows:
# - Scripts: 'unsafe-inline' (Required for Plotly/interactive charts)
# - Styles: 'unsafe-inline' (Required for styling)
# - Images: 'self' and data: URIs (For embedded plots)
# Blocks everything else (default-src 'none') to prevent XSS/exfiltration
CSP_META_CONTENT = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:;"


class ScriptHasher(HTMLParser):
    """
    HTML Parser to extract inline scripts for hashing and external script sources.
    This enables strict CSP generation by avoiding 'unsafe-inline' for scripts.
    """

    def __init__(self):
        super().__init__()
        self.hashes = set()
        self.sources = set()
        self.in_script = False
        self.current_script = []
        self.current_script_has_src = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
            self.current_script = []
            self.current_script_has_src = False

            # Check for src attribute
            attrs_dict = dict(attrs)
            src = attrs_dict.get("src")
            if src:
                self.current_script_has_src = True
                # Add domain to sources
                try:
                    parsed = urlparse(src)
                    if parsed.scheme and parsed.netloc:
                        origin = f"{parsed.scheme}://{parsed.netloc}"
                        self.sources.add(origin)
                    elif src.startswith("//"):
                        # Protocol relative, assume https
                        parsed = urlparse(f"https:{src}")
                        if parsed.netloc:
                            origin = f"https://{parsed.netloc}"
                            self.sources.add(origin)
                except Exception:
                    pass

    def handle_endtag(self, tag):
        if tag == "script":
            if self.in_script and not self.current_script_has_src:
                script_content = "".join(self.current_script)
                if script_content:
                    # Calculate SHA-256 hash
                    sha256_hash = hashlib.sha256(script_content.encode("utf-8")).digest()
                    base64_hash = base64.b64encode(sha256_hash).decode("utf-8")
                    self.hashes.add(f"'sha256-{base64_hash}'")

            self.in_script = False
            self.current_script = []
            self.current_script_has_src = False

    def handle_data(self, data):
        if self.in_script:
            self.current_script.append(data)


def redact_sensitive_data(
    data: Any, max_depth: int = 50, _current_depth: int = 0
) -> Any:
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
        return "***RECURSION LIMIT EXCEEDED***"

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
                        if (
                            key_lower == sensitive
                            or key_lower.endswith(f"_{sensitive}")
                            or key_lower.endswith(f"-{sensitive}")
                        ):
                            is_sensitive = True
                            break

                if is_sensitive:
                    redacted[key] = "***REDACTED***"
                else:
                    redacted[key] = redact_sensitive_data(
                        value, max_depth, _current_depth + 1
                    )
            else:
                redacted[key] = redact_sensitive_data(
                    value, max_depth, _current_depth + 1
                )
        return redacted

    elif isinstance(data, list):
        return [
            redact_sensitive_data(item, max_depth, _current_depth + 1) for item in data
        ]

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
                        if key_lower == sensitive or key_lower.endswith(
                            f"_{sensitive}"
                        ):
                            is_sensitive = True
                            break

                if is_sensitive:
                    redacted_params.append((key, "***REDACTED***"))
                else:
                    redacted_params.append((key, value))

            # Allow * in value (safe='*') to prevent encoding of ***REDACTED***
            new_query = urlencode(redacted_params, doseq=True, safe="*")
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
        raise ValueError(
            f"Identifier exceeds maximum length of {MAX_ID_LENGTH} characters"
        )

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
            key_bytes = key.encode("utf-8")
        else:
            key_bytes = key

        return hmac.new(key_bytes, data.encode("utf-8"), hashlib.sha256).hexdigest()

    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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
    keys_pattern = "|".join(re.escape(k) for k in SENSITIVE_KEYS)

    # Simple pattern for standard assignments (key=value, key: value) without internal spaces/commas in value
    # We handle quoted values specially

    # 1. Match quoted values: key="value with spaces"
    pattern_quoted = re.compile(
        r"(?i)\b(" + keys_pattern + r')\b\s*[:=]\s*(["\'])(.*?)\2', re.DOTALL
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

    # 1. Double quotes: "value" - handles escaped double quotes \" and truncated strings
    # We use \\.? to handle escaped characters, including if the string is truncated right after backslash
    pattern_double = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*[:=]\s*)(")((?:[^"\\]|\\.?)*)(?:"|$)',
        re.DOTALL,
    )
    text = pattern_double.sub(r'\1\2\1\3"***REDACTED***"', text)

    # 2. Single quotes: 'value' - handles escaped single quotes \' and truncated strings
    pattern_single = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r")\1(\s*[:=]\s*)(\')((?:[^\'\\]|\\.?)*)(?:'|$)",
        re.DOTALL,
    )
    # Note: Use plain string with ' for replacement to avoid double escaping issues
    text = pattern_single.sub(r"\1\2\1\3'***REDACTED***'", text)

    # For unquoted: Replace group 4 (value) with ***REDACTED***
    # Group 1: Optional Quote
    # Group 2: Key
    # Group 3: Separator with optional surrounding whitespace
    # Group 4: Value (varies based on separator)

    # 3. Unquoted values with '=' (strict whitespace termination)
    # Handles query params, shell-style (key=value next=val)
    # Stops at whitespace, comma, semicolon, braces/brackets
    pattern_unquoted_equals = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*=\s*)([^"\'\s,;}\]]+)', re.DOTALL
    )
    text = pattern_unquoted_equals.sub(r"\1\2\1\3***REDACTED***", text)

    # 4. Unquoted values with ':' (permissive space, strict delimiter termination)
    # Handles HTTP headers (Authorization: Bearer token), YAML-style, JSON-like (key: value with spaces)
    # Allows spaces but stops at newline, comma, semicolon, braces/brackets
    # Explicitly excludes newlines (\n\r) to prevent multi-line consumption
    # Uses lookahead (?=\S) to ensure value starts with non-whitespace, preventing
    # matches on just the space before a quoted value (e.g. "key": "value")
    pattern_unquoted_colon = re.compile(
        r'(?i)(["\']?)(' + keys_pattern + r')\1(\s*:\s*)(?=\S)([^"\'\n\r,;}\]]+)', re.DOTALL
    )
    text = pattern_unquoted_colon.sub(r"\1\2\1\3***REDACTED***", text)

    return text


def inject_csp_meta_tag(html_content: str) -> str:
    """
    Inject Content Security Policy (CSP) meta tag into HTML content.

    This ensures that all generated HTML files (dashboards, reports) enforce
    strict security controls, preventing XSS and data exfiltration.

    It calculates SHA-256 hashes for inline scripts to avoid 'unsafe-inline'.

    Args:
        html_content: The original HTML string.

    Returns:
        The HTML string with the CSP meta tag injected into the <head>.
    """
    if not html_content or not isinstance(html_content, str):
        return html_content

    # Try to calculate hashes for stricter CSP
    try:
        parser = ScriptHasher()
        parser.feed(html_content)

        script_srcs = ["'self'"]

        # Add hashes
        if parser.hashes:
            script_srcs.extend(sorted(list(parser.hashes)))

        # Add allowed sources (CDNs)
        if parser.sources:
            script_srcs.extend(sorted(list(parser.sources)))

        script_policy = " ".join(script_srcs)

        # If we have hashes or sources, use them. Otherwise fallback to unsafe-inline
        # (though ideally we should fail closed, but empty script_srcs means no scripts found,
        # so 'self' is fine).
        # Wait, if there ARE scripts but parser failed to find them (unlikely with this parser),
        # they would be blocked. This is good (Fail Closed).

        # Construct strict policy
        # Note: style-src 'unsafe-inline' is still needed for Plotly's inline styles
        csp_content = (
            f"default-src 'none'; "
            f"script-src {script_policy}; "
            f"style-src 'unsafe-inline'; "
            f"img-src 'self' data: https:;"
        )

    except Exception:
        # Fallback to legacy unsafe-inline if parsing fails
        csp_content = CSP_META_CONTENT

    # Prepare CSP tag
    csp_tag = (
        f'<meta http-equiv="Content-Security-Policy" content="{csp_content}">'
    )

    # Check if CSP is already present to avoid duplication
    if 'http-equiv="Content-Security-Policy"' in html_content:
        return html_content

    # Insert into <head>
    if "<head>" in html_content:
        # Insert after <head> tag
        return html_content.replace("<head>", f"<head>\n    {csp_tag}", 1)
    elif "<html>" in html_content:
        # No head, insert after html
        return html_content.replace(
            "<html>", f"<html>\n<head>\n    {csp_tag}\n</head>", 1
        )
    else:
        # No structure, prepend it
        return f"{csp_tag}\n{html_content}"
