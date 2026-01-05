## 2024-05-23 - Sensitive Data Leakage in Logs
**Vulnerability:** API request parameters and cache keys were being logged in plain text. This could potentially expose sensitive information like access tokens, passwords, or API keys if they were passed as parameters (even if the current implementation puts the main API key in headers, future usage or other auth tokens could be leaked).
**Learning:** Even if the primary authentication mechanism (headers) is secure, logging "all parameters" for debugging purposes is a common source of data leaks. Cache keys often include all parameters to ensure uniqueness, inadvertently becoming a leak vector.
**Prevention:**
1. Implemented `redact_sensitive_data` utility to scrub sensitive keys from dictionaries before logging.
2. Hashed cache keys in logs instead of printing the raw key containing parameters.
3. Applied redaction to the API client's request logging.

## 2024-05-23 - Credential Leakage in URL Logs
**Vulnerability:** URLs containing credentials (e.g., `https://user:password@host/`) were being logged in plain text by the API client. This exposed passwords in logs and error messages.
**Learning:** Standard URL logging does not automatically mask credentials embedded in the URL authority section. Any system accepting a base URL (which might include Basic Auth) must sanitize it before logging.
**Prevention:**
1. Implemented `redact_url` in `security_utils.py` to parse and mask passwords in URLs.
2. Updated `api_client.py` to use `redact_url` for all logging and error messages involving URLs.

## 2024-01-05 - Extended CSV Injection Vectors
**Vulnerability:** The standard list of CSV injection characters (`=`, `+`, `-`, `@`) was insufficient to prevent all forms of spreadsheet injection, specifically those exploiting `\t` (Tab) and `\r` (Carriage Return) to manipulate cell formatting or structure.
**Learning:** Defense in depth for CSV injection requires handling control characters that can be interpreted by spreadsheet software as delimiters or formatting instructions, not just formula triggers.
**Prevention:** Expanded `CSV_INJECTION_CHARS` in `export_utils.py` to include `\t` and `\r`, ensuring these are also escaped/sanitized during export.
