import pandas as pd
import numpy as np
from ivi_water.data_processor import DataProcessor


class TestDataProcessorOptimization:
    def setup_method(self):
        self.processor = DataProcessor()

        # Create sample data
        N = 1000
        self.df = pd.DataFrame({
            # Unsorted location_id
            "location_id": np.random.choice(["C", "A", "B"], N),
            "year": np.random.randint(2000, 2023, N),
            "season": np.random.choice(["winter", "monsoon", "summer"], N),
            "water_area_ha": np.random.uniform(0, 100, N),
            "water_body_count": np.random.randint(1, 20, N),
        })

    def test_calculate_water_trends_sorted_output(self):
        """Test calculate_water_trends returns sorted output (by group keys)."""
        result = self.processor.calculate_water_trends(
            self.df, group_by=["location_id", "season"]
        )

        # Check if sorted by location_id then season
        expected_sort = result.sort_values(
            ["location_id", "season"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected_sort)

        # Verify unique combinations
        combinations = result[["location_id", "season"]].drop_duplicates()
        assert len(combinations) == len(result)

        # Verify correctness of sorting explicitly
        assert result["location_id"].is_monotonic_increasing

    def test_create_seasonal_summary_sorted_output(self):
        """Test that create_seasonal_summary returns sorted output."""
        result = self.processor.create_seasonal_summary(
            self.df, location_level="location_id"
        )

        # Check if sorted by location_id then season
        expected_sort = result.sort_values(
            ["location_id", "season"]
        ).reset_index(drop=True)
        # Note: Depending on implementation details (reset_index),
        # indexes might differ, so we ignore index
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_sort
        )

        # Verify correctness of sorting explicitly
        assert result["location_id"].is_monotonic_increasing
