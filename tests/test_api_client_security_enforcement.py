"""
CoREStackClient Security Enforcement Tests
"""

import os
import pytest
from unittest.mock import patch
from ivi_water.api_client import CoREStackClient


class TestAPIClientSecurityEnforcement:

    @patch("socket.getaddrinfo")
    def test_https_url_allowed(self, mock_getaddrinfo):
        """Test that HTTPS URLs are allowed."""
        # Mock DNS to return a public IP
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        client = CoREStackClient(api_key="dummy", base_url="https://api.example.com")
        assert client.base_url == "https://api.example.com"

    @patch("socket.getaddrinfo")
    def test_localhost_http_allowed(self, mock_getaddrinfo, monkeypatch):
        """Test that HTTP is allowed for localhost (requires internal IP override)."""
        monkeypatch.setenv("CORE_ALLOW_INTERNAL_IPS", "1")
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 8000))]

        client = CoREStackClient(api_key="dummy", base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

        client = CoREStackClient(api_key="dummy", base_url="http://127.0.0.1:8000")
        assert client.base_url == "http://127.0.0.1:8000"

    def test_insecure_http_rejected(self):
        """Test that non-localhost HTTP URLs are rejected by default."""
        with pytest.raises(ValueError, match="Insecure connection"):
            CoREStackClient(api_key="dummy", base_url="http://api.example.com")

    @patch("socket.getaddrinfo")
    def test_insecure_http_override(self, mock_getaddrinfo, monkeypatch):
        """Test that environment variable allows insecure HTTP."""
        monkeypatch.setenv("CORE_ALLOW_INSECURE_HTTP", "1")
        # Mock DNS to return a public IP
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]

        client = CoREStackClient(api_key="dummy", base_url="http://api.example.com")
        assert client.base_url == "http://api.example.com"
