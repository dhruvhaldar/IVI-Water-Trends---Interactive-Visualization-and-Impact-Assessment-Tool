
import os
import pytest
from unittest.mock import MagicMock
from ivi_water.api_client import CoREStackClient

class TestAPISecurity:
    def test_cache_key_does_not_leak_secrets(self):
        """
        Verify that cache keys are hashed and do not contain sensitive parameters in plain text.
        """
        # Setup client
        os.environ['CORE_API_KEY'] = 'dummy-key'
        client = CoREStackClient()

        # Mock session request to ensure success and cache population
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        client.session.request = MagicMock(return_value=mock_response)

        # Sensitive parameters
        sensitive_token = "SECRET_TOKEN_XYZ"
        sensitive_params = {
            "token": sensitive_token,
            "api_secret": "do_not_leak_this"
        }

        # Make request
        client._make_request("secure-endpoint", sensitive_params, use_cache=True)

        # Check cache keys
        cache_keys = list(client._cache.keys())
        assert len(cache_keys) > 0, "Cache should not be empty"

        for key in cache_keys:
            # Verify no sensitive data in key
            assert sensitive_token not in key, f"Cache key leaked token: {key}"
            assert "do_not_leak_this" not in key, f"Cache key leaked secret: {key}"

            # Verify key looks hashed (should contain hex digest)
            # The key format is method_url_hash
            parts = key.split('_')
            key_hash = parts[-1]
            assert len(key_hash) == 64, f"Expected SHA256 hash length (64), got {len(key_hash)} in {key}"

    def test_cache_key_stability(self):
        """
        Verify that cache keys are stable (deterministic) for same input.
        """
        client = CoREStackClient(api_key="dummy")

        # Mock session
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        client.session.request = MagicMock(return_value=mock_response)

        params = {"a": 1, "b": 2}

        # Request 1
        client._make_request("endpoint", params, use_cache=True)
        key1 = list(client._cache.keys())[0]

        # Clear cache
        client.clear_cache()

        # Request 2 (same params)
        client._make_request("endpoint", params, use_cache=True)
        key2 = list(client._cache.keys())[0]

        assert key1 == key2, "Cache key should be deterministic"

    def test_cache_key_does_not_leak_url_credentials(self):
        """
        Verify that credentials in base_url are not leaked in cache keys.
        """
        # Setup client with credentials in base_url
        sensitive_pass = "super_secret_password"
        base_url = f"https://user:{sensitive_pass}@api.corestack.org/v1"
        client = CoREStackClient(api_key="dummy", base_url=base_url)

        # Mock session
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        client.session.request = MagicMock(return_value=mock_response)

        # Make request
        client._make_request("endpoint", {}, use_cache=True)

        # Check cache keys
        cache_keys = list(client._cache.keys())
        assert len(cache_keys) > 0, "Cache should not be empty"

        for key in cache_keys:
            # Verify no sensitive data in key
            assert sensitive_pass not in key, f"Cache key leaked URL credential: {key}"

            # Verify key structure: method_urlhash_paramshash
            parts = key.split('_')
            # method is GET (part 0)
            # url_hash (part 1)
            # params_hash (part 2)
            assert len(parts) == 3
            assert len(parts[1]) == 64, f"Expected hashed URL in key, got {parts[1]}"
