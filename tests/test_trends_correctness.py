import pandas as pd
import numpy as np
import pytest
from ivi_water.data_processor import DataProcessor

class TestTrendsCorrectness:
    def test_slope_calculation_edge_cases(self):
        processor = DataProcessor()

        # Create data for 3 locations
        # Loc1: Perfect linear trend (slope = 1)
        # Loc2: Constant water area (slope = 0, denominator = 0)
        # Loc3: Insufficient data (N=1) -> Should not be in output or handled
        # Loc4: Vertical line? (Not possible with years as x)
        # Loc5: Two points (N=2)

        data = {
            'location_id': [],
            'year': [],
            'season': [],
            'water_area_ha': []
        }

        # Loc1: 2020: 10, 2021: 11, 2022: 12. Slope = 1.
        for y, val in zip([2020, 2021, 2022], [10, 11, 12]):
            data['location_id'].append('Loc1')
            data['year'].append(y)
            data['season'].append('monsoon')
            data['water_area_ha'].append(val)

        # Loc2: 2020: 10, 2021: 10, 2022: 10. Slope = 0.
        for y, val in zip([2020, 2021, 2022], [10, 10, 10]):
            data['location_id'].append('Loc2')
            data['year'].append(y)
            data['season'].append('monsoon')
            data['water_area_ha'].append(val)

        # Loc3: 2020: 10. N=1.
        data['location_id'].append('Loc3')
        data['year'].append(2020)
        data['season'].append('monsoon')
        data['water_area_ha'].append(10)

        # Loc5: 2020: 10, 2022: 20. Slope = (20-10)/(2022-2020) = 5.
        data['location_id'].append('Loc5')
        data['year'].append(2020)
        data['season'].append('monsoon')
        data['water_area_ha'].append(10)
        data['location_id'].append('Loc5')
        data['year'].append(2022)
        data['season'].append('monsoon')
        data['water_area_ha'].append(20)

        df = pd.DataFrame(data)

        results = processor.calculate_water_trends(df, group_by=['location_id', 'season'])

        # Check Loc1
        res1 = results[results['location_id'] == 'Loc1'].iloc[0]
        assert np.isclose(res1['trend_slope_ha_per_year'], 1.0)
        assert res1['trend_quality'] == 'good'

        # Check Loc2
        res2 = results[results['location_id'] == 'Loc2'].iloc[0]
        assert np.isclose(res2['trend_slope_ha_per_year'], 0.0)
        # Depending on implementation, constant year might be handled.
        # My code sets trend_quality to "constant_year" if denominator is near 0.
        # Wait, for constant Y (water area), numerator is 0. Denominator is non-zero (variance of X/year is non-zero).
        # So slope should be 0.
        # Let's check logic: denominator = N * sum_xx - sum_x**2.
        # Loc2: Years 2020, 2021, 2022. Variance of X is > 0. Denominator > 0.
        # Numerator: N*sum_xy - sum_x*sum_y.
        # y is constant C. sum_xy = sum(x*C) = C*sum_x.
        # Num = N*C*sum_x - sum_x*(N*C) = 0.
        # So slope is 0. Correct. trend_quality should be 'good' (or whatever default is).
        assert res2['trend_quality'] == 'good'

        # Check Loc3
        # Should be filtered out or marked insufficient_data
        if 'Loc3' in results['location_id'].values:
            res3 = results[results['location_id'] == 'Loc3'].iloc[0]
            assert res3['trend_quality'] == 'insufficient_data'

        # Check Loc5
        res5 = results[results['location_id'] == 'Loc5'].iloc[0]
        assert np.isclose(res5['trend_slope_ha_per_year'], 5.0)
        assert res5['trend_quality'] == 'minimal_data' # N=2

    def test_constant_year_handling(self):
        # Case where denominator is 0 (all years are same)
        processor = DataProcessor()
        data = {
            'location_id': ['LocConst', 'LocConst'],
            'year': [2020, 2020], # Same year!
            'season': ['monsoon', 'monsoon'],
            'water_area_ha': [10, 20]
        }
        df = pd.DataFrame(data)
        results = processor.calculate_water_trends(df, group_by=['location_id', 'season'])

        res = results[results['location_id'] == 'LocConst'].iloc[0]
        assert res['trend_quality'] == 'constant_year'
        assert res['trend_slope_ha_per_year'] == 0.0
