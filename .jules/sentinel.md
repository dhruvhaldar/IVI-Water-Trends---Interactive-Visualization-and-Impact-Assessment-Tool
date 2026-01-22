## 2024-05-24 - SSRF Protection in API Clients
**Vulnerability:** The API client allowed setting a `base_url` pointing to internal/private IP addresses (including `localhost` and cloud metadata services like `169.254.169.254`), creating a Server-Side Request Forgery (SSRF) risk.
**Learning:** Simply checking the URL scheme (http/https) or string matching 'localhost' is insufficient. DNS resolution is required to inspect the actual target IP address.
**Prevention:** Implement a `validate_base_url` method that resolves the hostname using `socket.getaddrinfo` and checks the resulting IP object (using `ipaddress` library) against `is_private`, `is_loopback`, and `is_reserved`. Allow an explicit override (e.g., `CORE_ALLOW_INTERNAL_IPS`) for legitimate internal use cases.

## 2024-05-24 - Strict URL Scheme Validation
**Vulnerability:** The API client validated URLs by checking for `http` (insecurity) and resolving hostnames (SSRF), but failed to restrict the URL scheme itself. This allowed potentially dangerous schemes like `ftp://`, `file://`, or `javascript:` to bypass validation if they didn't trigger the specific HTTP or DNS checks.
**Learning:** Security checks based on exclusion (blocking 'http') are fragile. Always use an allowlist approach for critical parameters like URL schemes.
**Prevention:** Enforce a strict whitelist of allowed schemes (e.g., `{'https'}`) before performing other validations.

## 2024-05-25 - Inconsistent Security Controls in CLI vs Core
**Vulnerability:** While the core `DataProcessor` enforced file size limits to prevent DoS via memory exhaustion, the CLI bypassed this control by directly using `pd.read_csv` for some operations, creating an inconsistency where the same operation (loading data) was secure in one context but not another.
**Learning:** Security controls implemented in business logic (e.g., `DataProcessor`) must be encapsulated in reusable methods (like `load_csv_safe`) and strictly used by all interfaces (CLI, API, etc.). Ad-hoc implementations in entry points often miss these controls.
**Prevention:** Centralize sensitive operations (like file loading) into secure utility methods and ensure all entry points use them instead of raw library calls.

## 2024-05-25 - Inconsistent CSP Enforcement in HTML Exports
**Vulnerability:** While `save_figure` injected Content Security Policy (CSP) headers to prevent XSS, other export methods like `create_multi_location_dashboard` and `create_visualization_exports` generated HTML files directly (using `fig.write_html`), bypassing these security controls.
**Learning:** When multiple methods perform similar output generation (e.g., saving HTML), they often drift in security implementation. Ad-hoc injection of security headers is error-prone.
**Prevention:** Centralize the HTML generation and security header injection logic into a single utility function (e.g., `inject_csp_meta_tag`) and enforce its use across all export functions.

## 2024-05-25 - DoS via Unbounded API Response
**Vulnerability:** The API client used `requests.request` without `stream=True` and blindly called `response.json()`, causing the entire response body to be loaded into memory. This created a Denial of Service (DoS) vulnerability where a malicious or misconfigured server could exhaust the client's memory by sending a massive response.
**Learning:** Relying on default behavior of HTTP clients often leads to unsafe memory usage for untrusted inputs. Always assume external inputs can be arbitrarily large.
**Prevention:** Use `stream=True` and iterate over the response content in chunks. Enforce a strict maximum size limit (e.g., `CORE_API_MAX_RESPONSE_SIZE`) and abort the connection if the limit is exceeded.

## 2024-05-25 - DoS via Zip Bomb in CSV Loading
**Vulnerability:** The `load_csv_safe` method enforced a file size limit based on the file size on disk (`file_path.stat().st_size`). This was vulnerable to "Zip Bomb" (decompression bomb) attacks where a small compressed file (e.g., `.csv.gz`) expands to consume excessive memory upon loading by Pandas.
**Learning:** Checking file size on disk is insufficient for compressed files. Security controls must account for the *expanded* size of data in memory.
**Prevention:** Implement chunked reading (`pd.read_csv(..., chunksize=N)`) and monitor the cumulative memory usage of the chunks. If the expanded data size exceeds the limit, abort the process and release memory.
