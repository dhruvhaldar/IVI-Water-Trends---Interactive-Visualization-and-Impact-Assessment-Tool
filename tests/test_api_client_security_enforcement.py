"""
CoREStackClient Security Enforcement Tests
"""

import os
import pytest
from ivi_water.api_client import CoREStackClient

class TestAPIClientSecurityEnforcement:

    def test_https_url_allowed(self):
        """Test that HTTPS URLs are allowed."""
        client = CoREStackClient(api_key="dummy", base_url="https://api.example.com")
        assert client.base_url == "https://api.example.com"

    def test_localhost_http_allowed(self):
        """Test that HTTP is allowed for localhost."""
        client = CoREStackClient(api_key="dummy", base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

        client = CoREStackClient(api_key="dummy", base_url="http://127.0.0.1:8000")
        assert client.base_url == "http://127.0.0.1:8000"

    def test_insecure_http_rejected(self):
        """Test that non-localhost HTTP URLs are rejected by default."""
        with pytest.raises(ValueError, match="Insecure connection"):
            CoREStackClient(api_key="dummy", base_url="http://api.example.com")

    def test_insecure_http_override(self, monkeypatch):
        """Test that environment variable allows insecure HTTP."""
        monkeypatch.setenv('CORE_ALLOW_INSECURE_HTTP', '1')
        client = CoREStackClient(api_key="dummy", base_url="http://api.example.com")
        assert client.base_url == "http://api.example.com"
