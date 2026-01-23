
import unittest
import pandas as pd
from ivi_water.data_processor import DataProcessor

class TestSlopeCorrectness(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()

    def test_slope_calculation_zeros(self):
        # Create data where slope denominator might be zero (constant water area over time)
        # Denominator = N * sum(x^2) - sum(x)^2
        # If years are same (variance 0), denominator is 0. But here year is variable.
        # Wait, if we group by location/season, and a group has identical years (duplicates?), denominator is 0.
        # But we aggregated rows. If we have multiple rows for same year?
        # Standard slope formula assumes unique x or handles duplicates.
        # Variance of X = 0 means X is constant.

        # Case 1: Constant Year (should result in slope 0 or handling)
        df = pd.DataFrame({
            'location_id': ['L1', 'L1'],
            'year': [2020, 2020], # Constant year
            'season': ['monsoon', 'monsoon'],
            'water_area_ha': [10, 20]
        })

        # calculate_water_trends groups by location/season
        result = self.processor.calculate_water_trends(df)

        # Check slope is 0 (handled by code)
        self.assertEqual(result.iloc[0]['trend_slope_ha_per_year'], 0.0)
        self.assertEqual(result.iloc[0]['trend_quality'], 'constant_year')

    def test_slope_calculation_normal(self):
        # Case 2: Normal trend
        # y = 2x + 10 => x=1, y=12; x=2, y=14
        df = pd.DataFrame({
            'location_id': ['L2', 'L2'],
            'year': [2020, 2021],
            'season': ['monsoon', 'monsoon'],
            'water_area_ha': [100, 110] # Slope 10
        })

        result = self.processor.calculate_water_trends(df)
        self.assertAlmostEqual(result.iloc[0]['trend_slope_ha_per_year'], 10.0)

    def test_slope_calculation_division_by_zero_handling(self):
        # To specifically trigger division by zero in the vectorized op but not "constant_year" logic (if possible)
        # or just ensure robust handling.
        # If N=0 (filtered out earlier).
        # If N * sum(xx) == sum(x)^2. This means Var(X) = 0. Handled by constant_year check.

        # What if N=1?
        # N=1 => N*xx - x^2 = 1*x^2 - x^2 = 0.
        # calculate_water_trends filters data_points > 0.
        # But MIN_DATA_POINTS_FOR_TREND is 2.
        # Code:
        # mask_insuf = stats_df["data_points"] < MIN_DATA_POINTS_FOR_TREND
        # stats_df.loc[mask_insuf, "trend_quality"] = "insufficient_data"
        # stats_df.loc[mask_insuf, "trend_slope_ha_per_year"] = 0.0

        # So even if we get NaN/Inf from calculation, it should be overwritten by 0.0 for N < 2.

        df = pd.DataFrame({
            'location_id': ['L3'],
            'year': [2020],
            'season': ['monsoon'],
            'water_area_ha': [100]
        })
        result = self.processor.calculate_water_trends(df)

        # Since logic filters stats_df[data_points > 0], this row remains.
        # Then N=1. Denominator=0. Slope calc -> Inf/NaN.
        # Then fillna(0).
        # Then insufficient data overwrite.

        self.assertEqual(result.iloc[0]['trend_slope_ha_per_year'], 0.0)
        self.assertEqual(result.iloc[0]['trend_quality'], 'insufficient_data')

if __name__ == '__main__':
    unittest.main()
