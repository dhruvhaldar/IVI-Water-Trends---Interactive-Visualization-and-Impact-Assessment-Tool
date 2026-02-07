## 2024-05-23 - Regex Redaction Bypass
**Vulnerability:** Standard regex word boundaries `\b` treat `_` as a word character. This caused sensitive keys with underscore prefixes (e.g., `my_token` where `token` is sensitive) to bypass redaction because `_` does not trigger a word boundary.
**Learning:** `\b` is often insufficient for security redaction when keys might be part of snake_case identifiers.
**Prevention:** Use negative lookbehind `(?<![a-zA-Z0-9])` instead of `\b` to assert the start of a sensitive key, allowing separators like `_`, `-`, or `.` while avoiding matches within alphanumeric strings (e.g., `monkey`).
