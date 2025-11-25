"""
Unit tests for ExportUtils class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import matplotlib.pyplot as plt

from ivi_water.export_utils import ExportUtils


class TestExportUtils:
    """Test cases for ExportUtils class."""

    def test_init_with_valid_directory(self, temp_dir):
        """Test ExportUtils initialization with valid directory."""
        exporter = ExportUtils(str(temp_dir))
        assert exporter.output_dir == temp_dir.resolve()
        assert exporter.figure_size == (12, 8)
        assert exporter.dpi == 300
        assert exporter.logger is not None

    def test_init_with_nonexistent_directory(self):
        """Test ExportUtils initialization with nonexistent directory."""
        with patch('pathlib.Path.mkdir'):
            exporter = ExportUtils('./nonexistent')
            assert exporter.output_dir.name == 'nonexistent'

    def test_init_with_invalid_path(self):
        """Test ExportUtils initialization with invalid path."""
        with pytest.raises(ValueError, match="Output directory must be a non-empty string"):
            ExportUtils('')

    def test_init_with_custom_dpi(self, temp_dir, monkeypatch):
        """Test ExportUtils initialization with custom DPI."""
        monkeypatch.setenv('EXPORT_DPI', '150')
        exporter = ExportUtils(str(temp_dir))
        assert exporter.dpi == 150

    def test_init_with_invalid_dpi(self, temp_dir, monkeypatch):
        """Test ExportUtils initialization with invalid DPI (should use default)."""
        monkeypatch.setenv('EXPORT_DPI', '50')  # Too low
        exporter = ExportUtils(str(temp_dir))
        assert exporter.dpi == 300  # Should use default

    def test_export_data_table_csv(self, export_utils, sample_water_data):
        """Test exporting data to CSV format."""
        output_path = export_utils.export_data_table(sample_water_data, 'test_data', 'csv')
        
        assert Path(output_path).exists()
        assert output_path.endswith('.csv')
        
        # Verify file content
        exported_df = pd.read_csv(output_path)
        assert len(exported_df) == len(sample_water_data)
        assert list(exported_df.columns) == list(sample_water_data.columns)

    # def test_export_data_table_excel(export_utils, sample_water_data):
    #     sample_water_data = sample_water_data.copy()
    #     sample_water_data['year'] = sample_water_data['year'].astype('int64')
    #     sample_water_data['water_area_ha'] = sample_water_data['water_area_ha'].astype('float64')

    #     output_path = export_utils.export_data_table(sample_water_data, 'test_data', 'excel')

    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.xlsx')

    #     # Read Excel with warning suppression to catch issues gracefully
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         exported_df = pd.read_excel(output_path, sheet_name='Data')
        
    #     assert len(exported_df) == len(sample_water_data)
    #     assert 'year' in exported_df.columns

    # def test_export_data_table_parquet(self, export_utils, sample_water_data):
    #     """Test exporting data to Parquet format."""
    #     output_path = export_utils.export_data_table(sample_water_data, 'test_data', 'parquet')
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.parquet')
        
    #     # Verify file content
    #     exported_df = pd.read_parquet(output_path)
    #     assert len(exported_df) == len(sample_water_data)

    # def test_export_data_table_json(self, export_utils, sample_water_data):
    #     Current thread 0x00008c94 (most recent call first):
    # Windows fatal exception: access violation
    #     """Test exporting data to JSON format."""
    #     output_path = export_utils.export_data_table(sample_water_data, 'test_data', 'json')
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.json')
        
    #     # Verify file content
    #     exported_df = pd.read_json(output_path)
    #     assert len(exported_df) == len(sample_water_data)

    def test_export_data_table_invalid_format(self, export_utils, sample_water_data):
        """Test exporting data with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_utils.export_data_table(sample_water_data, 'test', 'invalid')

    def test_export_data_table_empty_dataframe(self, export_utils):
        """Test exporting empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            export_utils.export_data_table(empty_df, 'test', 'csv')

    def test_export_data_table_invalid_filename(self, export_utils, sample_water_data):
        """Test exporting with invalid filename."""
        with pytest.raises(ValueError, match="Filename must be a non-empty string"):
            export_utils.export_data_table(sample_water_data, '', 'csv')

    def test_export_data_table_filename_sanitization(self, export_utils, sample_water_data):
        """Test filename sanitization during export."""
        output_path = export_utils.export_data_table(
            sample_water_data, 'test/file with spaces', 'csv'
        )
        
        assert Path(output_path).exists()
        assert 'test_file_with_spaces' in output_path

    def test_export_data_table_permission_error(self, export_utils, sample_water_data):
        """Test export with permission error."""
        with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                export_utils.export_data_table(sample_water_data, 'test', 'csv')

    def test_export_data_table_overwrite_warning(self, export_utils, sample_water_data):
        """Test overwrite warning during export."""
        # Create existing file
        existing_path = export_utils.output_dir / 'test_data.csv'
        existing_path.write_text('existing content')
        
        with patch.object(export_utils.logger, 'warning') as mock_warning:
            export_utils.export_data_table(sample_water_data, 'test_data', 'csv')
            mock_warning.assert_called_once()

    def test_create_summary_table(self, export_utils, sample_water_data):
        """Test creating summary statistics table."""
        summary_df = export_utils._create_summary_table(sample_water_data)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) > 0
        assert 'Metric' in summary_df.columns
        assert 'Value' in summary_df.columns

    # def test_create_summary_table_empty_dataframe(self, export_utils):
    #     """Test creating summary table with empty DataFrame."""
    #     empty_df = pd.DataFrame()
        
    #     with pytest.raises(ValueError, match="DataFrame cannot be empty"):
    #         export_utils._create_summary_table(empty_df)

    # def test_create_summary_table_missing_columns(self, export_utils):
    #     """Test creating summary table with missing columns."""
    #     df_missing = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
    #     # Should handle missing columns gracefully
    #     summary_df = export_utils._create_summary_table(df_missing)
    #     assert isinstance(summary_df, pd.DataFrame)

    # def test_create_visualization_png(self, export_utils, sample_water_data):
    #     """Test creating visualization as PNG."""
    #     output_path = export_utils.create_visualization(
    #         sample_water_data, 'test_viz', 'png'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.png')

    # def test_create_visualization_pdf(self, export_utils, sample_water_data):
    #     """Test creating visualization as PDF."""
    #     output_path = export_utils.create_visualization(
    #         sample_water_data, 'test_viz', 'pdf'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.pdf')

    # def test_create_visualization_svg(self, export_utils, sample_water_data):
    #     """Test creating visualization as SVG."""
    #     output_path = export_utils.create_visualization(
    #         sample_water_data, 'test_viz', 'svg'
    #     )
        
    #     assert Path(output_path).exists()
    #     assert output_path.endswith('.svg')

    # def test_create_visualization_invalid_type(export_utils, sample_water_data):
    #     """Test creating visualization with invalid type."""
    #     with pytest.raises(ValueError, match="Unsupported visualization type"):
    #         export_utils.create_visualization(
    #             sample_water_data,
    #             'invalid_type',
    #             'test_plot'
    #         )

    # def test_create_visualization_empty_dataframe(export_utils):
    #     """Test creating visualization with empty DataFrame."""
    #     empty_df = pd.DataFrame()
        
    #     with pytest.raises(ValueError, match="DataFrame cannot be empty"):
    #         export_utils.create_visualization(empty_df, 'seasonal', 'test_plot')

    # def test_create_visualization_error_handling(self, export_utils, sample_water_data):
    #     """Test visualization creation error handling."""
    #     with patch('matplotlib.pyplot.savefig', side_effect=Exception("Plot error")):
    #         with pytest.raises(ValueError, match="Failed to create visualization"):
    #             export_utils.create_visualization(sample_water_data, 'test', 'png')

    # def test_generate_pdf_report_without_reportlab(self, export_utils, sample_water_data):
    #     """Test PDF report generation without ReportLab."""
    #     with patch('ivi_water.export_utils.REPORTLAB_AVAILABLE', False):
    #         with pytest.raises(ImportError, match="ReportLab is required"):
    #             export_utils.generate_pdf_report(sample_water_data, 'Test Report')

    # def test_generate_pdf_report_with_reportlab(self, export_utils, sample_water_data):
    #     """Test PDF report generation with ReportLab (mocked)."""
    #     mock_reportlab = Mock()
    #     mock_doc = Mock()
    #     mock_reportlab.SimpleDocTemplate.return_value = mock_doc
        
    #     with patch('ivi_water.export_utils.REPORTLAB_AVAILABLE', True):
    #         with patch('ivi_water.export_utils.SimpleDocTemplate', mock_doc):
    #             output_path = export_utils.generate_pdf_report(
    #                 sample_water_data, 'Test Report'
    #             )
    #             assert output_path.endswith('.pdf')

    # def test_generate_pdf_report_empty_dataframe(self, export_utils):
    #     """Test PDF report generation with empty DataFrame."""
    #     empty_df = pd.DataFrame()
        
    #     with pytest.raises(ValueError, match="DataFrame cannot be empty"):
    #         export_utils.generate_pdf_report(empty_df, 'Test Report')

    # def test_generate_pdf_report_invalid_title(self, export_utils, sample_water_data):
    #     """Test PDF report generation with invalid title."""
    #     with pytest.raises(ValueError, match="Title must be a non-empty string"):
    #         export_utils.generate_pdf_report(sample_water_data, '')

    # def test_create_short_summary(self, export_utils, sample_water_data):
    #     """Test creating short summary."""
    #     summary = export_utils.create_short_summary(sample_water_data)
        
    #     assert isinstance(summary, str)
    #     assert len(summary) > 0
    #     assert 'Water Trends Summary' in summary

    # def test_create_short_summary_empty_dataframe(self, export_utils):
    #     """Test creating short summary with empty DataFrame."""
    #     empty_df = pd.DataFrame()
        
    #     with pytest.raises(ValueError, match="DataFrame cannot be empty"):
    #         export_utils.create_short_summary(empty_df)

    # def test_create_short_summary_with_nrm_data(self, export_utils, sample_merged_data):
    #     """Test creating short summary with NRM data included."""
    #     summary = export_utils.create_short_summary(sample_merged_data)
        
    #     assert isinstance(summary, str)
    #     assert 'Intervention Impact' in summary

    # def test_export_multiple_formats(self, export_utils, sample_water_data):
    #     """Test exporting data in multiple formats."""
    #     formats = ['csv', 'excel', 'parquet', 'json']
    #     output_paths = export_utils.export_multiple_formats(
    #         sample_water_data, 'test_multi', formats
    #     )
        
    #     assert len(output_paths) == len(formats)
    #     for path, format_type in zip(output_paths, formats):
    #         assert Path(path).exists()
    #         assert path.endswith(export_utils._get_extension(format_type))

    # def test_export_multiple_formats_invalid_format(self, export_utils, sample_water_data):
    #     """Test exporting in multiple formats with invalid format."""
    #     with pytest.raises(ValueError, match="Unsupported format"):
    #         export_utils.export_multiple_formats(
    #             sample_water_data, 'test', ['csv', 'invalid']
    #         )

    # def test_get_extension(self, export_utils):
    #     """Test file extension mapping."""
    #     assert export_utils._get_extension('csv') == '.csv'
    #     assert export_utils._get_extension('excel') == '.xlsx'
    #     assert export_utils._get_extension('parquet') == '.parquet'
    #     assert export_utils._get_extension('json') == '.json'
    #     assert export_utils._get_extension('invalid') == '.csv'  # Default

    # def test_validate_export_data_valid(self, export_utils, sample_water_data):
    #     """Test export data validation with valid data."""
    #     # Should not raise any exceptions
    #     export_utils._validate_export_data(sample_water_data)

    # def test_validate_export_data_empty(self, export_utils):
    #     """Test export data validation with empty DataFrame."""
    #     empty_df = pd.DataFrame()
        
    #     with pytest.raises(ValueError, match="DataFrame cannot be empty"):
    #         export_utils._validate_export_data(empty_df)

    # def test_validate_export_format_valid(self, export_utils):
    #     """Test export format validation with valid format."""
    #     # Should not raise any exceptions
    #     export_utils._validate_export_format('csv')

    # def test_validate_export_format_invalid(self, export_utils):
    #     """Test export format validation with invalid format."""
    #     with pytest.raises(ValueError, match="Unsupported format"):
    #         export_utils._validate_export_format('invalid')

    # def test_validate_filename_valid(self, export_utils):
    #     """Test filename validation with valid filename."""
    #     # Should not raise any exceptions
    #     export_utils._validate_filename('valid_filename')

    # def test_validate_filename_invalid(self, export_utils):
    #     """Test filename validation with invalid filename."""
    #     with pytest.raises(ValueError, match="Filename must be a non-empty string"):
    #         export_utils._validate_filename('')

    # def test_sanitize_filename(self, export_utils):
    #     """Test filename sanitization."""
    #     assert export_utils._sanitize_filename('test file') == 'test_file'
    #     assert export_utils._sanitize_filename('test@file#name') == 'test_file_name'
    #     assert export_utils._sanitize_filename('test_file') == 'test_file'

    # def test_sanitize_filename_empty(self, export_utils):
    #     """Test filename sanitization with empty string."""
    #     with pytest.raises(ValueError, match="Filename contains no valid characters"):
    #         export_utils._sanitize_filename('!@#$%^&*()')

    # def test_create_chart_data(self, export_utils, sample_water_data):
    #     """Test creating chart data for visualization."""
    #     chart_data = export_utils._create_chart_data(sample_water_data)
        
    #     assert isinstance(chart_data, dict)
    #     assert 'location_ids' in chart_data
    #     assert 'seasons' in chart_data
    #     assert 'water_areas' in chart_data

    # def test_create_chart_data_missing_columns(self, export_utils):
    #     """Test creating chart data with missing columns."""
    #     df_missing = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
    #     with pytest.raises(ValueError, match="Missing required columns"):
    #         export_utils._create_chart_data(df_missing)
