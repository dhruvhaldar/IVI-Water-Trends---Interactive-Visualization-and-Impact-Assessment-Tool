"""
Security Utilities Module

This module provides helper functions for security-related tasks,
such as redacting sensitive information from logs.
"""

import re
import hmac
import hashlib
import base64
import os
import logging
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
    "cookie",
    "set-cookie",
    "session_id",
    "sid",
    "csrf_token",
    "xsrf_token",
    "aws_session_token",
    "session_token",
    "secret_access_key",
    "aws_secret_access_key",
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

# List of authentication schemes that might appear in Authorization headers
AUTH_PREFIXES = {
    "Bearer",
    "Basic",
    "Digest",
    "Negotiate",
    "HOBA",
    "Mutual",
    "AWS4-HMAC-SHA256",
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

# Allowed domains for scripts in CSP
DEFAULT_ALLOWED_SCRIPT_DOMAINS = {
    "cdn.plot.ly",
}

# --- Pre-compiled Regex Patterns (Performance Optimization) ---

# Construct regex components for sensitive keys
_KEYS_PATTERN = "|".join(re.escape(k) for k in SENSITIVE_KEYS)

# 1. Double quotes: "value" - handles escaped double quotes \" and truncated strings
_PATTERN_DOUBLE = re.compile(
    r'(?i)(["\']?)\b('
    + _KEYS_PATTERN
    + r')\1(\s*[:=]\s*)(")((?:[^"\\]|\\.?)*)(?:"|$)',
    re.DOTALL,
)

# 2. Single quotes: 'value' - handles escaped single quotes \' and truncated strings
_PATTERN_SINGLE = re.compile(
    r'(?i)(["\']?)\b('
    + _KEYS_PATTERN
    + r")\1(\s*[:=]\s*)(\')((?:[^\'\\]|\\.?)*)(?:'|$)",
    re.DOTALL,
)

# 3. Special handling for Authorization headers
_SCHEMES_PATTERN = "|".join(re.escape(s) for s in AUTH_PREFIXES)
_AUTH_KEYS_PATTERN = r"(?:Proxy-)?Authorization"

_PATTERN_AUTH = re.compile(
    r'(?i)(["\']?)\b('
    + _AUTH_KEYS_PATTERN
    + r')\1(\s*[:=]\s*)((?:'
    + _SCHEMES_PATTERN
    + r")\s+)([^\"'\s,;}\]]+)",
    re.DOTALL,
)

# 4. Unquoted values with '='
_PATTERN_UNQUOTED_EQUALS = re.compile(
    r'(?i)(["\']?)\b('
    + _KEYS_PATTERN
    + r')\1(\s*=\s*)([^"\'\s,;}\]]+)',
    re.DOTALL,
)

# 5. Unquoted values with ':'
_PATTERN_UNQUOTED_COLON = re.compile(
    r'(?i)(["\']?)\b('
    + _KEYS_PATTERN
    + r')\1(\s*:\s*)(?=\S)([^"\'\n\r,;}\]]+)',
    re.DOTALL,
)

# --- Terminal Sanitization Patterns ---

# Remove ANSI escape sequences (7-bit C1 ANSI sequences)
_ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Remove unsafe control characters (0-31 except 9,10,13; and 127)
# \t(9), \n(10), \r(13) are safe to keep (or rather, escape)
# Range \x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F
_UNSAFE_CONTROL_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')


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
        is_fallback = False

        # If no password found but @ present and no scheme, try fallback parsing
        if not parsed.password and "@" in url and "://" not in url:
            try:
                # Prepend // to treat as netloc-relative URL
                parsed_fb = urlparse("//" + url)
                if parsed_fb.password:
                    parsed = parsed_fb
                    is_fallback = True
            except Exception:
                pass

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

        # Handle dummy scheme cleanup if fallback was used
        result = urlunparse(parsed)
        if is_fallback and result.startswith("//") and not url.startswith("//"):
            result = result[2:]

        return result
    except Exception:
        # If parsing fails, return a placeholder to prevent leaking secrets
        # contained in malformed URLs (e.g. invalid ports)
        return "<REDACTION FAILED>"


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
        # Use repr() to escape control characters (like newlines) in the error message
        # to prevent log injection if the exception is logged.
        safe_id_repr = repr(clean_id)
        raise ValueError(
            f"Invalid identifier {safe_id_repr}. "
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

    # Apply redaction using pre-compiled regexes
    text = _PATTERN_DOUBLE.sub(r'\1\2\1\3"***REDACTED***"', text)
    text = _PATTERN_SINGLE.sub(r"\1\2\1\3'***REDACTED***'", text)
    text = _PATTERN_AUTH.sub(r"\1\2_IVIPROTECTED_\1\3\4***REDACTED***", text)
    text = _PATTERN_UNQUOTED_EQUALS.sub(r"\1\2\1\3***REDACTED***", text)
    text = _PATTERN_UNQUOTED_COLON.sub(r"\1\2\1\3***REDACTED***", text)

    # Restore Authorization keys by removing the protection suffix
    text = text.replace("_IVIPROTECTED_", "")

    return text


def inject_csp_meta_tag(html_content: str) -> str:
    """
    Inject Content Security Policy (CSP) meta tag into HTML content.

    This ensures that all generated HTML files (dashboards, reports) enforce
    strict security controls, preventing XSS and data exfiltration.

    It calculates SHA-256 hashes for inline scripts to avoid 'unsafe-inline'.
    It restricts external script sources to a safe whitelist.

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
            # Enforce whitelist for external script sources
            allowed_domains = DEFAULT_ALLOWED_SCRIPT_DOMAINS.copy()

            # Allow user to extend whitelist via environment variable
            env_allowed = os.getenv("IVI_CSP_ALLOWED_DOMAINS")
            if env_allowed:
                allowed_domains.update(
                    [d.strip() for d in env_allowed.split(",") if d.strip()]
                )

            logger = logging.getLogger(__name__)

            for source in parser.sources:
                # Extract hostname from source URL
                try:
                    # urlparse returns netloc (e.g., example.com:8080)
                    # We use hostname to ignore port (e.g., example.com)
                    # Use lower() for case-insensitive comparison
                    parsed_source = urlparse(source)
                    source_hostname = (parsed_source.hostname or "").lower()

                    if source_hostname in allowed_domains:
                        script_srcs.append(source)
                    else:
                        logger.warning(
                            f"Blocked unauthorized script source in CSP: {source_hostname} ({source}). "
                            "To allow, set IVI_CSP_ALLOWED_DOMAINS environment variable."
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse script source {source}: {e}")

        script_policy = " ".join(script_srcs)

        # Construct strict policy
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


def sanitize_for_terminal(text: str) -> str:
    """
    Sanitize text for safe terminal output.

    Removes ANSI escape sequences and other control characters to prevent
    terminal injection attacks (e.g. hiding output, spoofing, or executing commands
    in vulnerable terminals).

    Safe control characters (newline, tab, carriage return) are escaped to their
    literal representation (e.g. \\n) to preserve visibility while preventing execution.

    Args:
        text: Input text.

    Returns:
        Sanitized string safe for printing to terminal.
    """
    if not isinstance(text, str):
        return str(text)

    # Remove ANSI escape sequences
    text = _ANSI_ESCAPE_PATTERN.sub('', text)

    # Remove unsafe control characters
    text = _UNSAFE_CONTROL_PATTERN.sub('', text)

    # Escape safe control characters to prevent injection/formatting issues
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    return text
