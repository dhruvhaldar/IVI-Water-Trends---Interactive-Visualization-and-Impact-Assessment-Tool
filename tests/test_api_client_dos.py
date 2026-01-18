import unittest
from unittest.mock import MagicMock, patch, ANY
import json
import os
from requests.exceptions import RequestException
from ivi_water.api_client import CoREStackClient

class TestApiClientDoS(unittest.TestCase):
    def test_request_uses_stream(self):
        """
        Verify that _make_request uses stream=True.
        """
        client = CoREStackClient(api_key="test")

        with patch.object(client.session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Mock iter_content to return empty to avoid errors
            mock_response.iter_content.return_value = [b'{}']
            mock_response.headers = {}
            mock_request.return_value = mock_response

            client._make_request("test-endpoint")

            # Check arguments passed to request
            args, kwargs = mock_request.call_args
            self.assertTrue(kwargs.get('stream', False), "stream=True must be set")

    def test_response_size_limit_header(self):
        """
        Verify that requests are rejected if Content-Length exceeds limit.
        """
        # Set small limit for testing
        with patch.dict(os.environ, {'CORE_API_MAX_RESPONSE_SIZE': '100'}):
            client = CoREStackClient(api_key="test")
            self.assertEqual(client.max_response_size, 100)

            with patch.object(client.session, 'request') as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {'Content-Length': '200'}
                mock_request.return_value = mock_response

                # Expect RequestException because _make_request wraps exceptions
                with self.assertRaisesRegex(RequestException, "Response too large"):
                    client._make_request("test-endpoint")

    def test_response_size_limit_streaming(self):
        """
        Verify that requests are rejected if streamed content exceeds limit.
        """
        # Set small limit for testing (10 bytes)
        with patch.dict(os.environ, {'CORE_API_MAX_RESPONSE_SIZE': '10'}):
            client = CoREStackClient(api_key="test")

            with patch.object(client.session, 'request') as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {} # No Content-Length
                # Mock iter_content to yield chunks that exceed limit
                mock_response.iter_content.return_value = [b'12345', b'678901'] # 11 bytes total
                mock_request.return_value = mock_response

                with self.assertRaisesRegex(RequestException, "Response size exceeded limit"):
                    client._make_request("test-endpoint")

    def test_normal_response(self):
        """
        Verify that normal responses are processed correctly.
        """
        client = CoREStackClient(api_key="test")
        test_data = {"key": "value"}
        json_data = json.dumps(test_data).encode('utf-8')

        with patch.object(client.session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {'Content-Length': str(len(json_data))}
            mock_response.iter_content.return_value = [json_data]
            mock_request.return_value = mock_response

            result = client._make_request("test-endpoint")
            self.assertEqual(result, test_data)

if __name__ == '__main__':
    unittest.main()
