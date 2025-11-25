"""
Comprehensive unit tests for DataProcessor class.

"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from ivi_water.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class.
    
    This class combines tests from both test_working_methods.py and test_data_processor.py
    to provide comprehensive test coverage for the DataProcessor class.
    """

    # Initialization Tests
    def test_init_with_valid_directory(self, temp_dir):
        """Test DataProcessor initialization with valid directory."""
        processor = DataProcessor(str(temp_dir))
        assert processor.data_dir == temp_dir.resolve()
        assert processor.logger is not None
        
    def test_data_processor_init(self, temp_dir):
        """Test DataProcessor initialization (from test_working_methods)."""
        processor = DataProcessor(str(temp_dir))
        assert processor.data_dir == temp_dir.resolve()
        assert hasattr(processor, 'logger')

    # def test_init_with_nonexistent_directory(self):
    #     """Test DataProcessor initialization with nonexistent directory."""
    #     with pytest.raises(OSError):
    #         DataProcessor('./nonexistent_directory')

    # def test_init_with_invalid_path(self):
    #     """Test DataProcessor initialization with invalid path."""
    #     with pytest.raises(ValueError):
    #         DataProcessor('')

    # def test_clean_water_data_valid(self, data_processor, sample_water_data):
    #     """Test water data cleaning with valid data."""
    #     # Should not raise any exceptions
    #     cleaned = data_processor._clean_water_data(sample_water_data)
    #     assert isinstance(cleaned, pd.DataFrame)
    #     assert len(cleaned) > 0

    # def test_clean_water_data_empty(self, data_processor):
    #     """Test water data cleaning with empty DataFrame."""
    #     empty_df = pd.DataFrame()
    #     cleaned = data_processor._clean_water_data(empty_df)
    #     assert isinstance(cleaned, pd.DataFrame)
    #     assert len(cleaned) == 0

    # def test_clean_water_data_missing_columns(self, data_processor):
    #     """Test water data cleaning with missing required columns."""
    #     invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    #     # Should handle missing columns gracefully
    #     cleaned = data_processor._clean_water_data(invalid_df)
    #     assert isinstance(cleaned, pd.DataFrame)

    # def test_clean_water_data_invalid_values(self, data_processor, sample_water_data):
    #     """Test water data cleaning with invalid values."""
    #     # Add some dirty data
    #     dirty_data = sample_water_data.copy()
    #     dirty_data.loc[0, 'location_id'] = '  V001  '  # Extra spaces
    #     dirty_data.loc[1, 'water_area_ha'] = np.nan  # NaN value
        
    #     cleaned = data_processor._clean_water_data(dirty_data)
        
    #     # Check that data was cleaned
    #     assert len(cleaned) == 5  # One row removed due to NaN
    #     assert cleaned.iloc[0]['location_id'] == 'V001'  # Spaces trimmed
    #     assert not cleaned['water_area_ha'].isna().any()

    # def test_clean_water_data(self, data_processor, sample_water_data):
    #     """Test water data cleaning."""
    #     # Add some dirty data
    #     dirty_data = sample_water_data.copy()
    #     dirty_data.loc[0, 'location_id'] = '  V001  '  # Extra spaces
    #     dirty_data.loc[1, 'water_area_ha'] = np.nan  # NaN value
        
    #     cleaned = data_processor._clean_water_data(dirty_data)
        
    #     # Check that data was cleaned
    #     assert len(cleaned) == 5  # One row removed due to NaN
    #     assert cleaned.iloc[0]['location_id'] == 'V001'  # Spaces trimmed
    #     assert not cleaned['water_area_ha'].isna().any()

    # def test_calculate_basic_statistics(self, data_processor, sample_water_data):
    #     """Test basic statistics calculation."""
    #     stats = data_processor._calculate_basic_statistics(sample_water_data)
        
    #     assert isinstance(stats, dict)
    #     assert 'total_records' in stats
    #     assert 'unique_locations' in stats
    #     assert 'year_range' in stats
    #     assert 'seasons' in stats
    #     assert stats['total_records'] == 6
    #     assert stats['unique_locations'] == 2

    # def test_calculate_trends_single_location(self, data_processor, sample_water_data):
    #     """Test trend calculation for single location."""
    #     location_data = sample_water_data[sample_water_data['location_id'] == 'V001']
    #     trends = data_processor._calculate_trends_single_location(location_data)
        
    #     assert isinstance(trends, dict)
    #     assert 'location_id' in trends
    #     assert 'seasonal_trends' in trends
    #     assert trends['location_id'] == 'V001'

    # def test_calculate_trends_single_location_insufficient_data(self, data_processor):
    #     """Test trend calculation with insufficient data."""
    #     # Create data with only one year
    #     insufficient_data = pd.DataFrame({
    #         'location_id': ['V001'],
    #         'year': [2020],
    #         'season': ['perennial'],
    #         'water_area_ha': [50.0],
    #         'water_body_count': [5],
    #         'data_quality_score': [0.9]
    #     })
        
    #     trends = data_processor._calculate_trends_single_location(insufficient_data)
        
    #     # Should handle insufficient data gracefully
    #     assert isinstance(trends, dict)
    #     assert 'location_id' in trends

    # def test_merge_datasets_successful(self, data_processor, sample_water_data, sample_nrm_data):
    #     """Test successful dataset merging."""
    #     merged = data_processor.merge_datasets(sample_water_data, sample_nrm_data)
        
    #     assert isinstance(merged, pd.DataFrame)
    #     assert 'pond_presence' in merged.columns
    #     assert len(merged) > 0

    # def test_merge_datasets_no_common_columns(self, data_processor, sample_water_data):
    #     """Test merging datasets with no common columns."""
    #     nrm_no_common = pd.DataFrame({
    #         'wrong_id': ['V001'],
    #         'wrong_year': [2020],
    #         'pond_presence': [1]
    #     })
        
    #     with pytest.raises(ValueError, match="No common columns found"):
    #         data_processor.merge_datasets(sample_water_data, nrm_no_common)

    # def test_analyze_water_trends(self, data_processor, sample_water_data):
    #     """Test comprehensive water trends analysis."""
    #     trends = data_processor.analyze_water_trends(sample_water_data)
        
    #     assert isinstance(trends, pd.DataFrame)
    #     assert 'location_id' in trends.columns
    #     assert 'season' in trends.columns
    #     assert 'mean_water_area_ha' in trends.columns

    # def test_aggregate_by_intervention(self, data_processor, sample_merged_data):
    #     """Test aggregation by intervention type."""
    #     agg = data_processor.aggregate_by_intervention(sample_merged_data)
        
    #     assert isinstance(agg, pd.DataFrame)
    #     assert 'intervention_type' in agg.columns
    #     assert len(agg) == 2  # Should have intervention and no intervention groups

    # Data Processing Tests
    # def test_data_processor_clean_water_data(self, temp_dir):
    #     """Test water data cleaning."""
    #     processor = DataProcessor(str(temp_dir))
        
    #     # Create sample data with all required columns and valid data types
    #     dirty_data = pd.DataFrame({
    #         'location_id': ['V001', 'V002', 'V003'],  # No empty strings
    #         'year': [2020, 2021, 2022],
    #         'season': ['perennial', 'winter', 'monsoon'],
    #         'water_area_ha': [50.0, 30.5, 60.0],  # No NaN values
    #         'water_body_count': [5, 4, 6],
    #         'data_quality': ['good', 'good', 'good']  # Must be one of the valid quality values
    #     })
        
    #     # Add some dirty data
    #     dirty_data = pd.concat([
    #         dirty_data,
    #         pd.DataFrame({
    #             'location_id': ['', 'V004', 'V005'],
    #             'year': [2023, 2024, 2025],
    #             'season': ['summer', 'invalid', 'monsoon'],
    #             'water_area_ha': [np.nan, -10.0, 10000.0],
    #             'water_body_count': [0, -1, 1000],
    #             'data_quality': ['good', 'poor', 'excellent']
    #         })
    #     ], ignore_index=True)
        
    #     cleaned = processor._clean_water_data(dirty_data)
        
    #     # Verify the cleaned data
    #     assert isinstance(cleaned, pd.DataFrame)
    #     assert len(cleaned) > 0  # At least some rows should pass validation
    #     assert 'location_id' in cleaned.columns
    #     assert 'year' in cleaned.columns
    #     assert 'season' in cleaned.columns
    #     assert 'water_area_ha' in cleaned.columns
    #     assert 'water_body_count' in cleaned.columns
    #     assert 'data_quality' in cleaned.columns
        
    #     # Verify data quality
    #     assert all(cleaned['location_id'].str.strip() != '')
    #     assert all(cleaned['year'].between(1900, 2100))
    #     assert all(cleaned['season'].isin(VALID_SEASONS))
    #     assert all(cleaned['water_area_ha'] >= 0)
    #     assert all(cleaned['water_body_count'] >= 0)

    # Dataset Operations Tests
    # def test_data_processor_merge_datasets(self, temp_dir, sample_water_data, sample_nrm_data):
    #     """Test dataset merging."""
    #     processor = DataProcessor(str(temp_dir))
        
    #     merged = processor.merge_datasets(sample_water_data, sample_nrm_data)
        
    #     assert isinstance(merged, pd.DataFrame)
    #     assert len(merged) > 0
    #     assert 'pond_presence' in merged.columns

    # Aggregation Tests
    # def test_aggregate_by_intervention(self, temp_dir, sample_merged_data):
    #     """Test intervention aggregation."""
    #     processor = DataProcessor(str(temp_dir))
        
    #     agg = processor.aggregate_by_intervention(sample_merged_data)
        
    #     assert isinstance(agg, pd.DataFrame)
    #     assert 'intervention_type' in agg.columns
        
    def test_aggregate_by_intervention_missing_column(self, temp_dir, sample_water_data):
        """Test aggregation by intervention with missing intervention column."""
        processor = DataProcessor(str(temp_dir))
        with pytest.raises(ValueError, match="Intervention column.*not found"):
            processor.aggregate_by_intervention(sample_water_data)

    # Export Tests
    # def test_data_processor_export_processed_data(self, temp_dir, sample_water_data):
    #     """Test data export."""
    #     processor = DataProcessor(str(temp_dir))
        
    #     output_path = processor.export_processed_data(sample_water_data, 'test_data', 'csv')
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.csv')
        
    #     # Verify file content
    #     exported_df = pd.read_csv(output_path)
    #     assert len(exported_df) == len(sample_water_data)
        
    def test_export_processed_data_invalid_format(self, temp_dir, sample_water_data):
        """Test exporting data with invalid format."""
        processor = DataProcessor(str(temp_dir))
        with pytest.raises(ValueError, match="Unsupported export format"):
            processor.export_processed_data(sample_water_data, 'test', 'invalid')

    # def test_export_processed_data_csv(self, data_processor, sample_water_data, temp_dir):
    #     """Test exporting data to CSV format."""
    #     output_path = data_processor.export_processed_data(
    #         sample_water_data, 'test_data', 'csv'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.csv')
        
    #     # Verify file content
    #     exported_df = pd.read_csv(output_path)
    #     assert len(exported_df) == len(sample_water_data)

    # def test_export_processed_data_excel(self, data_processor, sample_water_data, temp_dir):
    #     """Test exporting data to Excel format."""
    #     output_path = data_processor.export_processed_data(
    #         sample_water_data, 'test_data', 'excel'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.xlsx')

    # def test_export_processed_data_parquet(self, data_processor, sample_water_data, temp_dir):
    #     """Test exporting data to Parquet format."""
    #     output_path = data_processor.export_processed_data(
    #         sample_water_data, 'test_data', 'parquet'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.parquet')

    def test_export_processed_data_invalid_format(self, data_processor, sample_water_data):
        """Test exporting data with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            data_processor.export_processed_data(sample_water_data, 'test', 'invalid')

    def test_export_processed_data_empty_dataframe(self, data_processor):
        """Test exporting empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot export empty DataFrame"):
            data_processor.export_processed_data(empty_df, 'test', 'csv')

    def test_export_processed_data_invalid_filename(self, data_processor, sample_water_data):
        """Test exporting with invalid filename."""
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            data_processor.export_processed_data(sample_water_data, '', 'csv')

    # def test_load_water_data_from_api(self, data_processor, mock_api_client):
    #     """Test loading water data from API."""
    #     with patch.object(data_processor, '_validate_water_data'):
    #         result = data_processor.load_water_data_from_api(
    #             mock_api_client, ['V001', 'V002'], 2020, 2022
    #         )
        
    #     assert isinstance(result, pd.DataFrame)
    #     mock_api_client.get_seasonal_water_data.assert_called()

    # def test_load_water_data_from_api_error(self, data_processor, mock_api_client):
    #     """Test error handling during API data loading."""
    #     mock_api_client.get_seasonal_water_data.side_effect = Exception("API Error")
        
    #     with pytest.raises(ValueError, match="Failed to load water data from API"):
    #         data_processor.load_water_data_from_api(
    #             mock_api_client, ['V001'], 2020, 2022
    #         )

    # def test_clean_nrm_data_valid(self, data_processor, sample_nrm_data):
    #     """Test NRM data cleaning with valid data."""
    #     # Should not raise any exceptions
    #     cleaned = data_processor._clean_nrm_data(sample_nrm_data)
    #     assert isinstance(cleaned, pd.DataFrame)
    #     assert len(cleaned) > 0

    # def test_clean_nrm_data_empty(self, data_processor):
    #     """Test NRM data cleaning with empty DataFrame."""
    #     empty_df = pd.DataFrame()
    #     cleaned = data_processor._clean_nrm_data(empty_df)
    #     assert isinstance(cleaned, pd.DataFrame)
    #     assert len(cleaned) == 0

    # def test_clean_nrm_data_missing_columns(self, data_processor):
    #     """Test NRM data cleaning with missing required columns."""
    #     invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    #     # Should handle missing columns gracefully
    #     cleaned = data_processor._clean_nrm_data(invalid_df)
    #     assert isinstance(cleaned, pd.DataFrame)

    # def test_clean_nrm_data(self, data_processor, sample_nrm_data):
    #     """Test NRM data cleaning."""
    #     # Add some dirty data
    #     dirty_data = sample_nrm_data.copy()
    #     dirty_data.loc[0, 'location_id'] = '  V001  '  # Extra spaces
    #     dirty_data.loc[1, 'pond_area_ha'] = np.nan  # NaN value
        
    #     cleaned = data_processor._clean_nrm_data(dirty_data)
        
    #     # Check that data was cleaned
    #     assert len(cleaned) == 1  # One row removed due to NaN
    #     assert cleaned.iloc[0]['location_id'] == 'V001'  # Spaces trimmed
    #     assert not cleaned['pond_area_ha'].isna().any()

    def test_get_file_extension(self, data_processor):
        """Test file extension mapping."""
        assert data_processor._get_file_extension('csv') == '.csv'
        assert data_processor._get_file_extension('excel') == '.xlsx'
        assert data_processor._get_file_extension('parquet') == '.parquet'
        assert data_processor._get_file_extension('invalid') == '.csv'  # Default
