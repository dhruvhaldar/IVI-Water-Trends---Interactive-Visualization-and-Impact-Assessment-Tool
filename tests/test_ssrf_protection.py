import os
import pytest
from unittest.mock import patch, MagicMock
from ivi_water.api_client import CoREStackClient


class TestSSRFProtection:
    """Test Suite for SSRF Protection in CoREStackClient."""

    def setup_method(self):
        # Reset env vars
        if "CORE_ALLOW_INTERNAL_IPS" in os.environ:
            del os.environ["CORE_ALLOW_INTERNAL_IPS"]
        if "CORE_API_KEY" not in os.environ:
            os.environ["CORE_API_KEY"] = "test-key"

    @patch("socket.getaddrinfo")
    def test_blocks_localhost_ip(self, mock_getaddrinfo):
        """Test that connecting to localhost IP is blocked."""
        # Mock DNS resolution to return 127.0.0.1
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]

        with pytest.raises(ValueError, match="Internal or private IP address"):
            CoREStackClient(base_url="https://127.0.0.1/v1")

    @patch("socket.getaddrinfo")
    def test_blocks_private_ip_range_10(self, mock_getaddrinfo):
        """Test that connecting to 10.x.x.x is blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("10.0.0.5", 443))]

        with pytest.raises(ValueError, match="Internal or private IP address"):
            CoREStackClient(base_url="https://10.0.0.5/v1")

    @patch("socket.getaddrinfo")
    def test_blocks_private_ip_range_192(self, mock_getaddrinfo):
        """Test that connecting to 192.168.x.x is blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("192.168.1.1", 443))]

        with pytest.raises(ValueError, match="Internal or private IP address"):
            CoREStackClient(base_url="https://192.168.1.1/v1")

    @patch("socket.getaddrinfo")
    def test_blocks_cloud_metadata_ip(self, mock_getaddrinfo):
        """Test that connecting to 169.254.169.254 is blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("169.254.169.254", 80))]

        # We use https here to bypass the http check and hit the SSRF check
        with pytest.raises(ValueError, match="Internal or private IP address"):
            CoREStackClient(base_url="https://169.254.169.254/latest/meta-data")

    @patch("socket.getaddrinfo")
    def test_allows_public_ip(self, mock_getaddrinfo):
        """Test that connecting to public IP is allowed."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("8.8.8.8", 443))]

        # Should not raise
        client = CoREStackClient(base_url="https://8.8.8.8/v1")
        assert client.base_url == "https://8.8.8.8/v1"

    @patch("socket.getaddrinfo")
    def test_allows_override(self, mock_getaddrinfo):
        """Test that CORE_ALLOW_INTERNAL_IPS overrides protection."""
        os.environ["CORE_ALLOW_INTERNAL_IPS"] = "true"
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]

        # Should not raise
        client = CoREStackClient(base_url="https://127.0.0.1/v1")
        assert client.base_url == "https://127.0.0.1/v1"
