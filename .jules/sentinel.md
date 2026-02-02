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

## 2024-05-25 - Zip Bomb Protection in Data Processor
**Vulnerability:** The `load_csv_safe` method in `DataProcessor` only checked the file size on disk (`st_size`) before loading. This was insufficient for compressed files (like `.csv.gz`), allowing a "Zip Bomb" (a small file on disk expanding to huge memory usage) to cause a Denial of Service (DoS) via memory exhaustion.
**Learning:** File size on disk is not a proxy for memory usage for compressed formats. Pandas `read_csv` transparently handles compression, hiding the expansion risk.
**Prevention:** Enforce chunked reading (`chunksize`) when loading untrusted CSVs. Iterate through chunks, track cumulative memory usage (using `df.memory_usage(deep=True)`), and abort if the limit is exceeded. Also, explicitly disable user-provided `chunksize` to ensure the security control cannot be bypassed.

## 2024-05-25 - Strict CSP with Hash-based Script Validation
**Vulnerability:** The application relied on `script-src 'unsafe-inline'` in its Content Security Policy (CSP) to support Plotly visualizations, effectively negating XSS protection against inline script injection.
**Learning:** Modern visualization libraries often rely on inline scripts, but allowing all inline scripts is dangerous. Using a custom HTML parser to extract and hash inline scripts allows generating a strict CSP that permits only the necessary scripts while blocking malicious injections.
**Prevention:** Implement a `ScriptHasher` (using `html.parser`) to scan generated HTML, calculate SHA-256 hashes of all inline scripts, and dynamically construct a `script-src` directive containing these hashes, enabling the removal of `'unsafe-inline'`.

## 2024-05-25 - Information Leakage in Truncated Logs
**Vulnerability:** The regex-based redaction utility (`redact_text_content`) required a closing quote to identify sensitive values (e.g., `key="value"`). If a log message was truncated (e.g., due to buffer limits), the closing quote would be missing, causing the regex to fail and the partial sensitive value to be leaked in plain text.
**Learning:** Security controls based on pattern matching must account for data stream interruptions. Regexes that strictly expect complete syntax (like closing quotes) fail securely in "open" contexts but fail insecurely in truncated contexts.
**Prevention:** Design redaction patterns to be resilient to truncation. Modify regexes to accept either the expected terminator (closing quote) OR the end of the string (`$`) as a valid match termination condition, ensuring that even partial secrets are redacted.

## 2024-05-25 - Terminal Injection Prevention
**Vulnerability:** The CLI tool printed data received from the API directly to the terminal without sanitization. If the API response contained ANSI escape sequences (e.g., from a compromised server or malicious data), it could hide output, spoof information, or potentially execute commands in vulnerable terminals.
**Learning:** Data from external sources (APIs, files) is untrusted even when displayed in a CLI. Terminal output is an injection vector just like HTML or SQL.
**Prevention:** Implement a `sanitize_for_terminal` utility that strips ANSI escape codes and unsafe control characters before printing any dynamic content to stdout.

## 2024-05-25 - Log Injection via Control Character Preservation
**Vulnerability:** The `sanitize_for_terminal` utility preserved newline (`\n`), carriage return (`\r`), and tab (`\t`) characters, allowing attackers to forge log entries or manipulate terminal output (Log Injection).
**Learning:** "Sanitizing" for terminal display often implies preserving formatting (like newlines), but for security logs, *any* control character can be used to spoof entries.
**Prevention:** Escape control characters (e.g., replace `\n` with literal `\\n`) in sanitization functions intended for logging or untrusted output, ensuring the content is visible but structurally inert.
