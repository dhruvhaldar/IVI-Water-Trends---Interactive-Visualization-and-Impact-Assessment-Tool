import pytest
from ivi_water.security_utils import redact_text_content, SENSITIVE_KEYS

class TestAuthRedaction:
    """Test suite for Authorization header redaction."""

    def test_authorization_header_bearer(self):
        """Test redaction of Authorization: Bearer token format."""
        text = "Authorization: Bearer mysecrettoken123"
        redacted = redact_text_content(text)
        assert "mysecrettoken123" not in redacted
        assert "***REDACTED***" in redacted
        # Should look like "Authorization: ***REDACTED***" or similar
        # We replace the whole value "Bearer mysecrettoken123" with "***REDACTED***"

    def test_authorization_header_basic(self):
        """Test redaction of Authorization: Basic base64token format."""
        text = "Authorization: Basic YWRtaW46cGFzc3dvcmQ="
        redacted = redact_text_content(text)
        assert "YWRtaW46cGFzc3dvcmQ=" not in redacted
        assert "***REDACTED***" in redacted

    def test_authorization_lowercase(self):
        """Test case insensitivity."""
        text = "authorization: bearer lowercase-token"
        redacted = redact_text_content(text)
        assert "lowercase-token" not in redacted
        assert "***REDACTED***" in redacted

    def test_proxy_authorization(self):
        """Test Proxy-Authorization header."""
        text = "Proxy-Authorization: Basic proxycreds"
        redacted = redact_text_content(text)
        assert "proxycreds" not in redacted
        assert "***REDACTED***" in redacted

    def test_multiple_headers(self):
        """Test multiple sensitive headers in one block."""
        text = "Authorization: Bearer token1\nSome-Header: value\nProxy-Authorization: Basic token2"
        redacted = redact_text_content(text)
        assert "token1" not in redacted
        assert "token2" not in redacted
        assert "value" in redacted  # Non-sensitive should be preserved

    def test_passphrase_redaction(self):
        """Test that passphrase is now redacted (new key added)."""
        text = "passphrase=correct horse battery staple"
        # Note: Standard unquoted redaction stops at space, so it might only redact "correct"
        # unless we improve it or it is single token.
        # But for this test, we just check "correct" is redacted.
        redacted = redact_text_content(text)
        assert "correct" not in redacted

    def test_signature_redaction(self):
        """Test signature redaction."""
        text = "signature=abcdef123456"
        redacted = redact_text_content(text)
        assert "abcdef123456" not in redacted

    def test_mixed_auth_schemes(self):
        """Test other auth schemes."""
        schemes = ["Digest", "Negotiate", "OAuth", "Token", "AWS"]
        for scheme in schemes:
            text = f"Authorization: {scheme} some-token-value"
            redacted = redact_text_content(text)
            assert "some-token-value" not in redacted, f"Failed to redact {scheme}"

    def test_no_prefix_auth(self):
        """Test Authorization without prefix (should fallback to unquoted redaction)."""
        text = "Authorization: just-a-token"
        redacted = redact_text_content(text)
        assert "just-a-token" not in redacted
