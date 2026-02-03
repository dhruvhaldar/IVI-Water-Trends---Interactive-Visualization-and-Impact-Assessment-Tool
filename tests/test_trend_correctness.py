import unittest
import pandas as pd
import numpy as np
from ivi_water.data_processor import DataProcessor

class TestTrendCorrectness(unittest.TestCase):
    def test_calculate_water_trends_slope(self):
        processor = DataProcessor()

        # Case 1: Perfect positive slope
        # Year: 2020, 2021, 2022
        # Area: 10, 20, 30
        # Slope should be 10.0
        df1 = pd.DataFrame({
            'location_id': ['Loc1'] * 3,
            'year': [2020, 2021, 2022],
            'season': ['perennial'] * 3,
            'water_area_ha': [10.0, 20.0, 30.0]
        })

        res1 = processor.calculate_water_trends(df1, group_by=['location_id', 'season'])
        slope1 = res1.iloc[0]['trend_slope_ha_per_year']
        self.assertAlmostEqual(slope1, 10.0, places=5)

        # Case 2: Constant (slope 0)
        df2 = pd.DataFrame({
            'location_id': ['Loc2'] * 3,
            'year': [2020, 2021, 2022],
            'season': ['perennial'] * 3,
            'water_area_ha': [15.0, 15.0, 15.0]
        })

        res2 = processor.calculate_water_trends(df2, group_by=['location_id', 'season'])
        slope2 = res2.iloc[0]['trend_slope_ha_per_year']
        self.assertAlmostEqual(slope2, 0.0, places=5)
        self.assertEqual(res2.iloc[0]['trend_quality'], 'good') # or whatever constant logic sets

        # Case 3: Constant year (denominator 0)
        # Should handle division by zero safely
        df3 = pd.DataFrame({
            'location_id': ['Loc3'] * 3,
            'year': [2020, 2020, 2020], # Same year
            'season': ['perennial'] * 3,
            'water_area_ha': [10.0, 20.0, 30.0]
        })

        res3 = processor.calculate_water_trends(df3, group_by=['location_id', 'season'])
        slope3 = res3.iloc[0]['trend_slope_ha_per_year']
        self.assertAlmostEqual(slope3, 0.0, places=5)
        self.assertEqual(res3.iloc[0]['trend_quality'], 'constant_year')

if __name__ == "__main__":
    unittest.main()
