import unittest
from unittest.mock import MagicMock
from ivi_water.api_client import CoREStackClient
from ivi_water.data_processor import DataProcessor


class TestBatchLimit(unittest.TestCase):
    def test_api_client_batch_limit(self):
        """Test that get_water_trends_summary enforces batch limit."""
        client = CoREStackClient(api_key="test")

        # Create a list of 101 location IDs
        location_ids = [f"V{i}" for i in range(101)]

        # Mock _make_request so we don't actually hit the API
        client._make_request = MagicMock()

        # Expect ValueError when calling with > 100 IDs
        with self.assertRaisesRegex(ValueError, "Batch size exceeds maximum limit"):
            client.get_water_trends_summary(location_ids, 2020, 2021)

    def test_data_processor_batch_limit(self):
        """Test that load_water_data_from_api enforces batch limit."""
        processor = DataProcessor()
        client = MagicMock()

        # Create a list of 101 location IDs
        location_ids = [f"V{i}" for i in range(101)]

        # Expect ValueError when calling with > 100 IDs
        with self.assertRaisesRegex(ValueError, "Batch size exceeds maximum limit"):
            processor.load_water_data_from_api(client, location_ids, 2020, 2021)


if __name__ == "__main__":
    unittest.main()
