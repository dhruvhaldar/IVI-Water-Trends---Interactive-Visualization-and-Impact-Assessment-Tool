
import pandas as pd
import numpy as np
import pytest
from ivi_water.data_processor import DataProcessor, WATER_DATA_COLUMNS

class TestDataProcessorCleaning:
    @pytest.fixture
    def processor(self):
        return DataProcessor()

    def test_clean_water_data_valid(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001'],
            'year': [2020],
            'season': ['winter'],
            'water_area_ha': [50.0],
            'water_body_count': [5],
            'data_quality': ['good']
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['location_id'] == 'V001'

    def test_clean_water_data_invalid_year(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001', 'V002'],
            'year': [1800, 2020],
            'season': ['winter', 'winter'],
            'water_area_ha': [50.0, 50.0],
            'water_body_count': [5, 5]
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['year'] == 2020

    def test_clean_water_data_invalid_season(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001', 'V002'],
            'year': [2020, 2020],
            'season': ['invalid', 'winter'],
            'water_area_ha': [50.0, 50.0],
            'water_body_count': [5, 5]
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['season'] == 'winter'

    def test_clean_water_data_missing_data(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001', 'V002', 'V003'],
            'year': [2020, np.nan, 2020],
            'season': ['winter', 'winter', np.nan],
            'water_area_ha': [50.0, 50.0, 50.0],
            'water_body_count': [5, 5, 5]
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['location_id'] == 'V001'

    def test_clean_water_data_invalid_area(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001', 'V002'],
            'year': [2020, 2020],
            'season': ['winter', 'winter'],
            'water_area_ha': [-10.0, 50.0],
            'water_body_count': [5, 5]
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['water_area_ha'] == 50.0

    def test_clean_water_data_negative_count(self, processor):
        df = pd.DataFrame({
            'location_id': ['V001'],
            'year': [2020],
            'season': ['winter'],
            'water_area_ha': [50.0],
            'water_body_count': [-5]
        })
        cleaned = processor._clean_water_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['water_body_count'] == 0
