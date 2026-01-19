import pytest
import os
from unittest.mock import patch
from ivi_water.api_client import CoREStackClient


class TestSchemeValidation:
    """Test Suite for URL Scheme Validation in CoREStackClient."""

    def setup_method(self):
        os.environ["CORE_API_KEY"] = "test-key"
        # Ensure insecure http is off by default
        if "CORE_ALLOW_INSECURE_HTTP" in os.environ:
            del os.environ["CORE_ALLOW_INSECURE_HTTP"]
        if "CORE_ALLOW_INTERNAL_IPS" in os.environ:
            del os.environ["CORE_ALLOW_INTERNAL_IPS"]

    @patch("socket.getaddrinfo")
    def test_blocks_ftp(self, mock_getaddrinfo):
        """Verify that FTP scheme is blocked."""
        # Mock DNS resolution for example.com
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 21))]

        with pytest.raises(ValueError, match="Invalid URL scheme 'ftp'"):
            CoREStackClient(base_url="ftp://example.com/v1")

    @patch("socket.getaddrinfo")
    def test_blocks_javascript(self, mock_getaddrinfo):
        """Verify that javascript scheme is blocked."""
        # Note: If DNS resolution fails, it will fail with "Could not resolve hostname",
        # but the scheme check happens BEFORE DNS resolution.
        # We assume hostname="" resolves to localhost in this mock to ensure scheme check is hit first.
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]

        with pytest.raises(ValueError, match="Invalid URL scheme 'javascript'"):
            CoREStackClient(base_url="javascript:alert(1)")

    @patch("socket.getaddrinfo")
    def test_blocks_file(self, mock_getaddrinfo):
        """Verify that file scheme is blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]

        with pytest.raises(ValueError, match="Invalid URL scheme 'file'"):
            CoREStackClient(base_url="file:///etc/passwd")

    @patch("socket.getaddrinfo")
    def test_allows_https(self, mock_getaddrinfo):
        """Verify that HTTPS is allowed."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        client = CoREStackClient(base_url="https://example.com/v1")
        assert client.base_url == "https://example.com/v1"

    @patch("socket.getaddrinfo")
    def test_allows_http_localhost_with_internal_override(self, mock_getaddrinfo):
        """
        Verify that HTTP is allowed for localhost IF internal IPs are also allowed.
        Because localhost resolves to 127.0.0.1, it triggers SSRF protection unless overridden.
        """
        os.environ["CORE_ALLOW_INTERNAL_IPS"] = "true"
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]
        client = CoREStackClient(base_url="http://localhost/v1")
        assert client.base_url == "http://localhost/v1"

    @patch("socket.getaddrinfo")
    def test_blocks_http_external(self, mock_getaddrinfo):
        """Verify that HTTP is blocked for external hosts by default."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]

        with pytest.raises(
            ValueError, match="Insecure connection: API base URL .* uses HTTP"
        ):
            CoREStackClient(base_url="http://example.com/v1")
