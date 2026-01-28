
import logging
import unittest
from unittest.mock import MagicMock
from ivi_water.data_processor import DataProcessor
from ivi_water.security_utils import sanitize_for_terminal

# Malicious ID with ANSI escape sequence (Red color)
MALICIOUS_ID = "V001\x1b[31m"
SANITIZED_ID = sanitize_for_terminal(MALICIOUS_ID)

class TestDataProcessorSecurity(unittest.TestCase):
    def test_log_injection(self):
        """
        Verify that user inputs (location IDs) are sanitized before being logged to prevent Terminal Injection.
        """
        # Setup capturing logger
        logger = logging.getLogger("ivi_water.data_processor")
        logger.setLevel(logging.INFO)

        from io import StringIO
        capture = StringIO()
        handler = logging.StreamHandler(capture)
        logger.addHandler(handler)

        try:
            processor = DataProcessor()

            # Mock API client to raise exception for the malicious ID
            mock_client = MagicMock()
            def side_effect(loc, *args):
                if loc == MALICIOUS_ID:
                    raise ValueError("Invalid ID")
                return {}

            mock_client.get_seasonal_water_data.side_effect = side_effect

            # Call the method
            try:
                processor.load_water_data_from_api(
                    mock_client, [MALICIOUS_ID], 2020, 2021
                )
            except ValueError:
                # Expected failure because the only location failed
                pass

            # Check logs
            log_output = capture.getvalue()

            # The raw escape sequence should NOT be in the logs
            self.assertNotIn("\x1b[", log_output, "ANSI escape sequence found in logs!")

            # The sanitized ID SHOULD be in the logs
            self.assertIn(SANITIZED_ID, log_output, "Sanitized ID not found in logs")

        finally:
            logger.removeHandler(handler)

    def test_log_sanitization_success_path(self):
        """
        Verify that even successful operations log sanitized IDs.
        """
        # Setup capturing logger
        logger = logging.getLogger("ivi_water.data_processor")
        logger.setLevel(logging.DEBUG)  # Enable debug logs

        from io import StringIO
        capture = StringIO()
        handler = logging.StreamHandler(capture)
        logger.addHandler(handler)

        try:
            processor = DataProcessor()

            # Mock API client to return valid data
            mock_client = MagicMock()
            mock_client.get_seasonal_water_data.return_value = {
                "timeseries": [
                    {"year": 2020, "seasons": {"monsoon": {"area_ha": 10, "count": 1}}}
                ]
            }

            # Use malicious ID that is technically valid for API client mock but unsafe for terminal
            processor.load_water_data_from_api(
                mock_client, [MALICIOUS_ID], 2020, 2021
            )

            log_output = capture.getvalue()

            # Check debug logs
            self.assertNotIn("\x1b[", log_output, "ANSI escape sequence found in success logs!")
            self.assertIn(SANITIZED_ID, log_output, "Sanitized ID not found in success logs")

        finally:
            logger.removeHandler(handler)

if __name__ == "__main__":
    unittest.main()
