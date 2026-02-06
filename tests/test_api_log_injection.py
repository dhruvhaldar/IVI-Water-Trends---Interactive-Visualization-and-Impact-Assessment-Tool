import logging
import unittest
from unittest.mock import MagicMock, patch
from ivi_water.api_client import CoREStackClient


class TestApiLogInjection(unittest.TestCase):
    """Test suite for API Client Log Injection prevention."""

    def setUp(self):
        self.logger_mock = MagicMock()

    def test_log_injection_in_url(self):
        """Test that ANSI escape codes in URL are sanitized in logs."""
        # Inject malicious ANSI code in endpoint
        malicious_endpoint = "v1/spatial-units/\x1b[31mMALICIOUS\x1b[0m"

        with patch("ivi_water.api_client.requests.Session") as mock_session:
            # Setup client
            client = CoREStackClient(api_key="test-key")
            # Replace client logger with mock
            client.logger = self.logger_mock

            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_response.encoding = "utf-8"
            mock_response.iter_content.return_value = [b'{"data": []}']

            session_instance = mock_session.return_value
            session_instance.request.return_value = mock_response
            client.session = session_instance

            # Call _make_request directly
            try:
                client._make_request(endpoint=malicious_endpoint)
            except Exception as e:
                # We don't expect an exception here, but if one occurs, catch it
                pass

            # Check debug logs
            found_ansi_in_log = False
            for call in self.logger_mock.debug.call_args_list:
                args, _ = call
                log_msg = args[0]
                print(f"Log: {repr(log_msg)}")
                if "\x1b" in log_msg:
                    found_ansi_in_log = True
                    break

            # Assert that ANSI codes are NOT found in logs (test fails if they are found)
            self.assertFalse(
                found_ansi_in_log,
                "ANSI escape codes found in logs! Log injection vulnerability detected.",
            )


if __name__ == "__main__":
    unittest.main()
