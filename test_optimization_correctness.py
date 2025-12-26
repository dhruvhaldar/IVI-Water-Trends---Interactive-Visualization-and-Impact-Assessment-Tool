
import pandas as pd
import numpy as np
import pytest
from ivi_water.data_processor import DataProcessor

def test_trend_calculation_correctness():
    processor = DataProcessor()

    # Case 1: Perfect linear trend
    df1 = pd.DataFrame({
        'location_id': ['LOC1'] * 3,
        'season': ['season1'] * 3,
        'year': [2020, 2021, 2022],
        'water_area_ha': [100.0, 110.0, 120.0]
    })

    trends1 = processor.calculate_water_trends(df1, group_by=['location_id', 'season'])
    row1 = trends1.iloc[0]
    print(f"Case 1 Slope: {row1['trend_slope_ha_per_year']} (Expected: 10.0)")
    assert np.isclose(row1['trend_slope_ha_per_year'], 10.0)
    assert row1['trend_quality'] == 'good'
    assert row1['data_points'] == 3

    # Case 2: Constant
    df2 = pd.DataFrame({
        'location_id': ['LOC2'] * 3,
        'season': ['season1'] * 3,
        'year': [2020, 2021, 2022],
        'water_area_ha': [100.0, 100.0, 100.0]
    })

    trends2 = processor.calculate_water_trends(df2, group_by=['location_id', 'season'])
    row2 = trends2.iloc[0]
    print(f"Case 2 Slope: {row2['trend_slope_ha_per_year']} (Expected: 0.0)")
    assert np.isclose(row2['trend_slope_ha_per_year'], 0.0)
    assert row2['trend_quality'] == 'good'

    # Case 3: 2 points
    df3 = pd.DataFrame({
        'location_id': ['LOC3'] * 2,
        'season': ['season1'] * 2,
        'year': [2020, 2022],
        'water_area_ha': [100.0, 120.0]
    })

    trends3 = processor.calculate_water_trends(df3, group_by=['location_id', 'season'])
    row3 = trends3.iloc[0]
    print(f"Case 3 Slope: {row3['trend_slope_ha_per_year']} (Expected: 10.0)")
    assert np.isclose(row3['trend_slope_ha_per_year'], 10.0)
    assert row3['trend_quality'] == 'minimal_data'

    # Case 4: Insufficient data (1 point)
    df4 = pd.DataFrame({
        'location_id': ['LOC4'],
        'season': ['season1'],
        'year': [2020],
        'water_area_ha': [100.0]
    })

    trends4 = processor.calculate_water_trends(df4, group_by=['location_id', 'season'])
    row4 = trends4.iloc[0]
    print(f"Case 4 Slope: {row4['trend_slope_ha_per_year']} (Expected: 0.0)")
    assert np.isclose(row4['trend_slope_ha_per_year'], 0.0)
    assert row4['trend_quality'] == 'insufficient_data'

    # Case 5: Constant Year (Should be constant_year quality)
    df5 = pd.DataFrame({
        'location_id': ['LOC5'] * 2,
        'season': ['season1'] * 2,
        'year': [2020, 2020],
        'water_area_ha': [100.0, 120.0]
    })

    trends5 = processor.calculate_water_trends(df5, group_by=['location_id', 'season'])
    row5 = trends5.iloc[0]
    print(f"Case 5 Slope: {row5['trend_slope_ha_per_year']} (Expected: 0.0)")
    # Slope is undefined/infinite, handled as 0.0 in code?
    # Logic: denominator = 0 -> slope = 0.
    assert np.isclose(row5['trend_slope_ha_per_year'], 0.0)
    assert row5['trend_quality'] == 'constant_year'

    # Case 6: Mixed groups
    df_mixed = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    trends_mixed = processor.calculate_water_trends(df_mixed, group_by=['location_id', 'season'])
    print(f"Mixed groups count: {len(trends_mixed)} (Expected: 5)")
    assert len(trends_mixed) == 5

if __name__ == "__main__":
    test_trend_calculation_correctness()
