import logging
import io
import unittest
from click.testing import CliRunner
from ivi_water.cli import cli


class TestCliLogInjection(unittest.TestCase):
    def setUp(self):
        # Reset logging handlers to avoid duplicates
        logging.getLogger().handlers = []

    def test_fetch_water_data_log_injection(self):
        # Capture logs
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)

        # Attach to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(ch)

        try:
            runner = CliRunner()

            # Payload with newline
            payload = "V001\n[CRITICAL] Fake Log Entry"
            escaped_payload = "V001\\n[CRITICAL] Fake Log Entry"

            # Run the command
            result = runner.invoke(
                cli,
                [
                    "fetch-water-data",
                    "--locations",
                    payload,
                    "--start-year",
                    "2020",
                    "--end-year",
                    "2021",
                ],
                env={"CORE_API_KEY": "dummy"},
            )

            log_contents = log_capture_string.getvalue()

            # Assert that the newline is escaped
            self.assertIn(escaped_payload, log_contents)
            self.assertNotIn("\n[CRITICAL] Fake Log Entry", log_contents)

        finally:
            root_logger.removeHandler(ch)

    def test_get_spatial_units_log_injection(self):
        # Capture logs
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(ch)

        try:
            runner = CliRunner()

            # Payload with newline
            payload = "State\n[CRITICAL] Fake Log"
            escaped_payload = "State\\n[CRITICAL] Fake Log"

            # Use correct command name 'get-spatial-units'
            result = runner.invoke(
                cli,
                ["get-spatial-units", "--state", payload],
                env={"CORE_API_KEY": "dummy"},
            )

            log_contents = log_capture_string.getvalue()

            if escaped_payload not in log_contents:
                print(f"\nLOG CONTENTS: {log_contents}")
                print(f"RESULT OUTPUT: {result.output}")

            self.assertIn(escaped_payload, log_contents)
            self.assertNotIn("\n[CRITICAL] Fake Log", log_contents)

        finally:
            root_logger.removeHandler(ch)


if __name__ == "__main__":
    unittest.main()
