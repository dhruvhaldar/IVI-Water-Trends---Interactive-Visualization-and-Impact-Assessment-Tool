import pandas as pd
import numpy as np
import pytest
from ivi_water.data_processor import DataProcessor

def test_calculate_water_trends_slope_correctness():
    processor = DataProcessor()

    # Case 1: Perfect linear trend
    # Year: 2020, 2021, 2022
    # Area: 100, 110, 120 -> Slope should be 10
    df = pd.DataFrame({
        "location_id": ["loc1"] * 3,
        "year": [2020, 2021, 2022],
        "season": ["monsoon"] * 3,
        "water_area_ha": [100, 110, 120]
    })

    trends = processor.calculate_water_trends(df, group_by=["location_id", "season"])

    assert len(trends) == 1
    slope = trends.iloc[0]["trend_slope_ha_per_year"]
    assert np.isclose(slope, 10.0)
    assert trends.iloc[0]["trend_quality"] == "good"

def test_calculate_water_trends_constant_area():
    processor = DataProcessor()

    # Case 2: Constant area (Slope 0)
    df = pd.DataFrame({
        "location_id": ["loc2"] * 3,
        "year": [2020, 2021, 2022],
        "season": ["winter"] * 3,
        "water_area_ha": [50, 50, 50]
    })

    trends = processor.calculate_water_trends(df, group_by=["location_id", "season"])

    slope = trends.iloc[0]["trend_slope_ha_per_year"]
    assert np.isclose(slope, 0.0)
    assert trends.iloc[0]["trend_quality"] == "good"

def test_calculate_water_trends_constant_year():
    processor = DataProcessor()

    # Case 3: Constant year (Division by zero in slope denominator)
    # Should result in slope 0 and quality 'constant_year'
    df = pd.DataFrame({
        "location_id": ["loc3"] * 3,
        "year": [2020, 2020, 2020],
        "season": ["summer"] * 3,
        "water_area_ha": [10, 20, 30]
    })

    trends = processor.calculate_water_trends(df, group_by=["location_id", "season"])

    slope = trends.iloc[0]["trend_slope_ha_per_year"]
    assert np.isclose(slope, 0.0)
    assert trends.iloc[0]["trend_quality"] == "constant_year"

def test_calculate_water_trends_insufficient_data():
    processor = DataProcessor()

    # Case 4: Single data point (Insufficient data)
    df = pd.DataFrame({
        "location_id": ["loc4"],
        "year": [2020],
        "season": ["monsoon"],
        "water_area_ha": [100]
    })

    trends = processor.calculate_water_trends(df, group_by=["location_id", "season"])

    slope = trends.iloc[0]["trend_slope_ha_per_year"]
    assert np.isclose(slope, 0.0)
    assert trends.iloc[0]["trend_quality"] == "insufficient_data"

def test_calculate_water_trends_minimal_data():
    processor = DataProcessor()

    # Case 5: Two data points (Minimal data)
    df = pd.DataFrame({
        "location_id": ["loc5"] * 2,
        "year": [2020, 2021],
        "season": ["monsoon"] * 2,
        "water_area_ha": [100, 110]
    })

    trends = processor.calculate_water_trends(df, group_by=["location_id", "season"])

    slope = trends.iloc[0]["trend_slope_ha_per_year"]
    assert np.isclose(slope, 10.0)
    assert trends.iloc[0]["trend_quality"] == "minimal_data"
