## 2024-05-23 - Enforcing HTTPS in API Clients
**Vulnerability:** The API client allowed configuring an `http://` base URL. Since authentication is done via `Authorization: Bearer <key>` header, using HTTP exposes the API key in plaintext to network observers.
**Learning:** Even if `requests` library supports HTTP, an API client handling sensitive credentials should fail-safe by enforcing HTTPS, especially when deployed in environments where network traffic might be intercepted.
**Prevention:** `CoREStackClient` now checks the protocol of `base_url` at initialization. It raises a `ValueError` for insecure HTTP unless the host is a loopback address or an override environment variable (`CORE_ALLOW_INSECURE_HTTP`) is explicitly set.
