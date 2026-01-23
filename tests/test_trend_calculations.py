
import pandas as pd
import numpy as np
import pytest
from ivi_water.data_processor import DataProcessor

class TestTrendCalculations:
    def setup_method(self):
        self.processor = DataProcessor()

    def test_trend_slope_simple(self):
        """Test simple positive linear trend"""
        df = pd.DataFrame({
            'location_id': ['Loc1'] * 3,
            'year': [2020, 2021, 2022],
            'season': ['all'] * 3,
            'water_area_ha': [10.0, 20.0, 30.0]
        })

        result = self.processor.calculate_water_trends(df, group_by=['location_id', 'season'])
        slope = result['trend_slope_ha_per_year'].iloc[0]

        # Expected slope: (30-10)/(2022-2020) = 10.0
        assert slope == 10.0

    def test_trend_slope_constant(self):
        """Test zero slope (constant values)"""
        df = pd.DataFrame({
            'location_id': ['Loc1'] * 3,
            'year': [2020, 2021, 2022],
            'season': ['all'] * 3,
            'water_area_ha': [10.0, 10.0, 10.0]
        })

        result = self.processor.calculate_water_trends(df, group_by=['location_id', 'season'])
        slope = result['trend_slope_ha_per_year'].iloc[0]
        quality = result['trend_quality'].iloc[0]

        assert slope == 0.0
        # Quality should be 'good' because we have enough data points and variance in years
        assert quality == 'good'

    def test_trend_slope_division_by_zero_variance(self):
        """Test case where all years are same (should not happen in valid time series but good for robustness)"""
        df = pd.DataFrame({
            'location_id': ['Loc1'] * 3,
            'year': [2020, 2020, 2020],
            'season': ['all'] * 3,
            'water_area_ha': [10.0, 20.0, 30.0]
        })

        result = self.processor.calculate_water_trends(df, group_by=['location_id', 'season'])
        slope = result['trend_slope_ha_per_year'].iloc[0]
        quality = result['trend_quality'].iloc[0]

        assert slope == 0.0
        assert quality == 'constant_year'

    def test_coefficient_of_variation(self):
        """Test CV calculation"""
        df = pd.DataFrame({
            'location_id': ['Loc1'] * 2,
            'year': [2020, 2021],
            'season': ['all'] * 2,
            'water_area_ha': [10.0, 30.0]
        })

        result = self.processor.calculate_water_trends(df, group_by=['location_id', 'season'])
        cv = result['coefficient_of_variation'].iloc[0]

        # Mean = 20.0
        # Std (ddof=1 default in pandas) = sqrt(((10-20)^2 + (30-20)^2)/1) = sqrt(200) = 14.1421356
        # CV = 14.1421356 / 20.0 = 0.70710678

        expected_cv = np.std([10.0, 30.0], ddof=1) / np.mean([10.0, 30.0])
        assert np.isclose(cv, expected_cv)

    def test_cv_division_by_zero(self):
        """Test CV when mean is 0"""
        df = pd.DataFrame({
            'location_id': ['Loc1'] * 2,
            'year': [2020, 2021],
            'season': ['all'] * 2,
            'water_area_ha': [0.0, 0.0]
        })

        result = self.processor.calculate_water_trends(df, group_by=['location_id', 'season'])
        cv = result['coefficient_of_variation'].iloc[0]

        assert cv == 0.0
