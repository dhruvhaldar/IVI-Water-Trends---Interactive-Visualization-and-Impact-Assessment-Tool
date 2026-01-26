import pandas as pd
import numpy as np
from ivi_water.data_processor import DataProcessor

def test_calculate_water_trends_basic():
    df = pd.DataFrame({
        'location_id': ['V001', 'V001', 'V001'],
        'year': [2020, 2021, 2022],
        'season': ['monsoon', 'monsoon', 'monsoon'],
        'water_area_ha': [100.0, 110.0, 105.0]
    })

    processor = DataProcessor()
    trends = processor.calculate_water_trends(df, ['location_id', 'season'])

    assert len(trends) == 1
    assert trends['trend_slope_ha_per_year'].iloc[0] == 2.5
    assert trends['mean_water_area_ha'].iloc[0] == 105.0
    assert trends['data_points'].iloc[0] == 3

def test_calculate_water_trends_constant():
    df = pd.DataFrame({
        'location_id': ['V001', 'V001'],
        'year': [2020, 2020],
        'season': ['monsoon', 'monsoon'],
        'water_area_ha': [100.0, 100.0]
    })

    processor = DataProcessor()
    trends = processor.calculate_water_trends(df, ['location_id', 'season'])

    # Logic handles constant year (denominator 0) -> slope 0, quality constant_year
    assert trends['trend_slope_ha_per_year'].iloc[0] == 0.0
    assert trends['trend_quality'].iloc[0] == 'constant_year'

def test_calculate_water_trends_insufficient():
    df = pd.DataFrame({
        'location_id': ['V001'],
        'year': [2020],
        'season': ['monsoon'],
        'water_area_ha': [100.0]
    })

    processor = DataProcessor()
    trends = processor.calculate_water_trends(df, ['location_id', 'season'])

    assert trends['trend_quality'].iloc[0] == 'insufficient_data'
    assert trends['trend_slope_ha_per_year'].iloc[0] == 0.0

def test_calculate_water_trends_cv():
    df = pd.DataFrame({
        'location_id': ['V001', 'V001'],
        'year': [2020, 2021],
        'season': ['monsoon', 'monsoon'],
        'water_area_ha': [100.0, 200.0]
    })

    processor = DataProcessor()
    trends = processor.calculate_water_trends(df, ['location_id', 'season'])

    mean = 150.0
    std = np.std([100.0, 200.0], ddof=1) # Pandas uses ddof=1 by default
    expected_cv = std / mean

    assert np.isclose(trends['coefficient_of_variation'].iloc[0], expected_cv)
