
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import pandas as pd
import stat
from ivi_water.data_processor import DataProcessor

class TestDoSProtection(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.test_file = 'test_large_file.csv'
        # Create a dummy file
        with open(self.test_file, 'w') as f:
            f.write("location_id,year,water_area_ha\n")
            f.write("V001,2020,10.5\n")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    @patch('pathlib.Path.stat')
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

    @patch('pathlib.Path.stat')
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

if __name__ == '__main__':
    unittest.main()
