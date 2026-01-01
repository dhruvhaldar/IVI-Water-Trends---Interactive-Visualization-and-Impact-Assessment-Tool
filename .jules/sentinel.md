## 2024-05-23 - Sensitive Data Leakage in Logs
**Vulnerability:** API request parameters and cache keys were being logged in plain text. This could potentially expose sensitive information like access tokens, passwords, or API keys if they were passed as parameters (even if the current implementation puts the main API key in headers, future usage or other auth tokens could be leaked).
**Learning:** Even if the primary authentication mechanism (headers) is secure, logging "all parameters" for debugging purposes is a common source of data leaks. Cache keys often include all parameters to ensure uniqueness, inadvertently becoming a leak vector.
**Prevention:**
1. Implemented `redact_sensitive_data` utility to scrub sensitive keys from dictionaries before logging.
2. Hashed cache keys in logs instead of printing the raw key containing parameters.
3. Applied redaction to the API client's request logging.
