import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import pandas as pd
import stat
from ivi_water.data_processor import DataProcessor
from requests.exceptions import RequestException
from ivi_water.api_client import CoREStackClient


class TestDoSProtection(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.test_file = "test_large_file.csv"
        # Create a dummy file
        with open(self.test_file, "w") as f:
            f.write("location_id,year,water_area_ha\n")
            f.write("V001,2020,10.5\n")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    @patch("pathlib.Path.stat")
    def test_load_large_file_raises_error(self, mock_stat):
        # Mock file size to be 201 MB
        large_size = 201 * 1024 * 1024

        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = large_size
        # st_mode for a regular file
        mock_stat_obj.st_mode = stat.S_IFREG
        mock_stat.return_value = mock_stat_obj

        # We expect it to FAIL initially because the code only warns
        # After fix, this should PASS
        with self.assertRaises(ValueError) as cm:
            self.processor.load_nrm_impact_data(self.test_file)

        self.assertIn("File size exceeds maximum limit", str(cm.exception))

    @patch("pathlib.Path.stat")
    def test_load_acceptable_file_size(self, mock_stat):
        # Mock file size to be 50 MB
        acceptable_size = 50 * 1024 * 1024

        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = acceptable_size
        mock_stat_obj.st_mode = stat.S_IFREG
        mock_stat.return_value = mock_stat_obj

        # Should not raise error
        try:
            df = self.processor.load_nrm_impact_data(self.test_file)
            self.assertFalse(df.empty)
        except ValueError as e:
            self.fail(f"load_nrm_impact_data raised ValueError unexpectedly: {e}")

    def test_zip_bomb_prevention(self):
        """Test that decompression bombs are detected even if file size is small."""
        # This test mocks the scenario where file on disk is small but expands to huge size

        # 1. Create a dummy compressed file (small on disk)
        import gzip

        compressed_file = "test_bomb.csv.gz"

        # Create content that compresses well (2MB uncompressed)
        # 2000 rows * 1024 chars = ~2MB
        content = ("A" * 1024 + "\n") * 2000
        with gzip.open(compressed_file, "wt") as f:
            f.write(content)

        # 2. Set strict limit (smaller than expanded size)
        # Set limit to 1MB
        # Note: We must patch os.environ for load_csv_safe to see the new limit
        with patch.dict(os.environ, {"MAX_FILE_SIZE_MB": "1"}):
            try:
                # 3. Attempt to load
                # The disk size check should pass (gzip is very small)
                # The chunked read check should fail (expands to > 1MB)
                with self.assertRaises(ValueError) as cm:
                    self.processor.load_csv_safe(compressed_file)

                self.assertIn("Decompression Bomb detected", str(cm.exception))
            finally:
                if os.path.exists(compressed_file):
                    os.remove(compressed_file)


class TestAPIResponseDoSProtection(unittest.TestCase):
    def test_large_response_dos_protection(self):
        """
        Verify that the client raises an error when the response size exceeds the limit.
        """
        # Set a small limit for testing (1KB)
        test_limit = 1024

        # Create a mock response that simulates a stream larger than the limit
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.status_code = 200

        # Create chunks that exceed the limit
        # 2 chunks of 600 bytes = 1200 bytes > 1024 bytes
        chunk_content = b"x" * 600
        # Use side_effect for iter_content to return a fresh iterator each time if needed
        mock_response.iter_content.return_value = iter([chunk_content, chunk_content])

        # Also mock 'content' property behavior for non-streaming access (legacy behavior)
        mock_response.content = b"x" * 1200
        mock_response.text = '{"key": "value"}'
        mock_response.json.return_value = {"key": "value"}
        # Mock encoding for clean logging
        mock_response.encoding = "utf-8"

        with patch.dict(os.environ, {"CORE_API_MAX_RESPONSE_SIZE": str(test_limit)}):
            client = CoREStackClient(api_key="test-key")

            # Patch the session.request to return our mock
            with patch.object(client.session, "request", return_value=mock_response):
                # Expect a RequestException (wrapping ValueError) due to size limit
                with self.assertRaises(RequestException) as cm:
                    client._make_request("test-endpoint", use_cache=False)

                self.assertIn("Response size exceeds limit", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
