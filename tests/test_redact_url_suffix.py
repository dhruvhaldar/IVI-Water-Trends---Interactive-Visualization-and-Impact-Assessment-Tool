
import pytest
from ivi_water.security_utils import redact_url

class TestRedactUrlSuffix:
    def test_redact_url_exact_match(self):
        """Test exact match for sensitive keys."""
        url = "http://example.com/api?api_key=secret"
        redacted = redact_url(url)
        assert "api_key=***REDACTED***" in redacted
        assert "secret" not in redacted

    def test_redact_url_underscore_suffix(self):
        """Test underscore suffix match."""
        url = "http://example.com/api?my_api_key=secret"
        redacted = redact_url(url)
        assert "my_api_key=***REDACTED***" in redacted
        assert "secret" not in redacted

    def test_redact_url_hyphen_suffix(self):
        """Test hyphen suffix match (The Fix)."""
        url = "http://example.com/api?my-api-key=secret"
        redacted = redact_url(url)
        assert "my-api-key=***REDACTED***" in redacted
        assert "secret" not in redacted

    def test_redact_url_no_suffix_match(self):
        """Test that keys without separator suffix do not match if not exact."""
        # 'public_key' is sensitive.
        # 'republic_key' ends with '_key' which is sensitive 'key' with separator.
        # But 'monkey' ends with 'key' WITHOUT separator.

        # 'key' is in SENSITIVE_KEYS.
        # 'monkey' should NOT be redacted.
        url = "http://example.com/api?monkey=banana"
        redacted = redact_url(url)
        assert "monkey=banana" in redacted

    def test_redact_url_mixed_separators(self):
        """Test mixed separators."""
        # 'client-secret' is sensitive.
        # 'my_client-secret' -> ends with '_client-secret'.
        url = "http://example.com/api?my_client-secret=top_secret"
        redacted = redact_url(url)
        assert "my_client-secret=***REDACTED***" in redacted
        assert "top_secret" not in redacted
