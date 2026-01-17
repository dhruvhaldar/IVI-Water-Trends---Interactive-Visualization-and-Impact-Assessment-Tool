## 2024-05-24 - SSRF Protection in API Clients
**Vulnerability:** The API client allowed setting a `base_url` pointing to internal/private IP addresses (including `localhost` and cloud metadata services like `169.254.169.254`), creating a Server-Side Request Forgery (SSRF) risk.
**Learning:** Simply checking the URL scheme (http/https) or string matching 'localhost' is insufficient. DNS resolution is required to inspect the actual target IP address.
**Prevention:** Implement a `validate_base_url` method that resolves the hostname using `socket.getaddrinfo` and checks the resulting IP object (using `ipaddress` library) against `is_private`, `is_loopback`, and `is_reserved`. Allow an explicit override (e.g., `CORE_ALLOW_INTERNAL_IPS`) for legitimate internal use cases.

## 2024-05-24 - Strict URL Scheme Validation
**Vulnerability:** The API client validated URLs by checking for `http` (insecurity) and resolving hostnames (SSRF), but failed to restrict the URL scheme itself. This allowed potentially dangerous schemes like `ftp://`, `file://`, or `javascript:` to bypass validation if they didn't trigger the specific HTTP or DNS checks.
**Learning:** Security checks based on exclusion (blocking 'http') are fragile. Always use an allowlist approach for critical parameters like URL schemes.
**Prevention:** Enforce a strict whitelist of allowed schemes (e.g., `{'https'}`) before performing other validations.
