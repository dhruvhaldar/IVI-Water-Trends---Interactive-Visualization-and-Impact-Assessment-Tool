import pytest
from unittest.mock import patch, MagicMock
from ivi_water.api_client import CoREStackClient
import socket
import os
from requests.exceptions import ConnectionError


class TestSSRFRedirectProtection:
    """
    Test suite to verify SSRF protection via redirects.
    """

    def setup_method(self):
        os.environ["CORE_API_KEY"] = "test-key"
        # Ensure internal IPs are blocked by default, cleaning up from other tests
        if "CORE_ALLOW_INTERNAL_IPS" in os.environ:
            del os.environ["CORE_ALLOW_INTERNAL_IPS"]

    @patch("socket.getaddrinfo")
    def test_blocks_redirect_to_private_ip(self, mock_getaddrinfo):
        """
        Verify that the client validates redirect targets and blocks unsafe ones.
        """

        # 1. Setup DNS: safe.com -> 8.8.8.8, internal.local -> 192.168.1.1
        def side_effect(host, *args, **kwargs):
            if host == "safe.com":
                return [(2, 1, 6, "", ("8.8.8.8", 443))]
            elif host == "internal.local":
                return [(2, 1, 6, "", ("192.168.1.1", 443))]
            return [(2, 1, 6, "", ("0.0.0.0", 0))]

        mock_getaddrinfo.side_effect = side_effect

        # 2. Initialize client with safe URL
        client = CoREStackClient(base_url="https://safe.com/v1")

        # 3. Simulate a redirect response
        mock_response = MagicMock()
        mock_response.is_redirect = True
        mock_response.status_code = 302
        mock_response.headers = {"Location": "https://internal.local/v1/secret"}
        mock_response.url = "https://safe.com/v1/data"

        # 4. Manually trigger response hooks
        hooks = client.session.hooks.get("response", [])
        if not isinstance(hooks, list):
            hooks = [hooks]

        # 5. Assert that ConnectionError is raised due to security check failure
        with pytest.raises(ConnectionError, match="Security check failed for redirect"):
            for hook in hooks:
                hook(mock_response)

    @patch("socket.getaddrinfo")
    def test_allows_safe_redirect(self, mock_getaddrinfo):
        """
        Verify that safe redirects are allowed.
        """
        # 1. Setup DNS: all safe
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("8.8.8.8", 443))]

        client = CoREStackClient(base_url="https://safe.com/v1")

        mock_response = MagicMock()
        mock_response.is_redirect = True
        mock_response.status_code = 302
        mock_response.headers = {"Location": "https://other-safe.com/v1/data"}
        mock_response.url = "https://safe.com/v1/data"

        hooks = client.session.hooks.get("response", [])
        if not isinstance(hooks, list):
            hooks = [hooks]

        # Should NOT raise exception
        for hook in hooks:
            hook(mock_response)
