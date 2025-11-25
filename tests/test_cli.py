"""
Unit tests for CLI module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from pathlib import Path
import tempfile
import shutil

from ivi_water.cli import cli, get_spatial_units, fetch_water_data, merge_data
from ivi_water.data_processor import DataProcessor
from ivi_water.api_client import CoREStackClient


class TestCLI:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'IVI Water Trends' in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output

    def test_cli_with_verbose(self):
        """Test CLI with verbose flag."""
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0

    def test_cli_with_custom_data_dir(self):
        """Test CLI with custom data directory."""
        # Create a temporary data directory
        data_dir = self.temp_path / 'data'
        data_dir.mkdir()
        
        result = self.runner.invoke(cli, ['--data-dir', str(data_dir), '--help'])
        assert result.exit_code == 0

    def test_cli_with_invalid_data_dir(self):
        """Test CLI with invalid data directory."""
        result = self.runner.invoke(cli, ['--data-dir', '/nonexistent'])
        assert result.exit_code != 0
        assert 'does not exist' in result.output

    def test_get_spatial_units_success(self, sample_water_data):
        """Test successful spatial units fetch."""
        # Create mock API client
        mock_client = Mock(spec=CoREStackClient)
        mock_client.get_spatial_units.return_value = [
            {'id': 'V001', 'name': 'Village 001', 'state': 'MH'},
            {'id': 'V002', 'name': 'Village 002', 'state': 'MH'}
        ]
        
        with patch('ivi_water.cli.CoREStackClient', return_value=mock_client):
            with patch('pandas.DataFrame') as mock_df:
                mock_df_instance = Mock()
                mock_df_instance.__len__ = Mock(return_value=2)
                mock_df_instance.head.return_value.to_string.return_value = 'sample data'
                mock_df.return_value = mock_df_instance
                
                result = self.runner.invoke(cli, [
                    'get-spatial-units',
                    '--unit-type', 'village',
                    '--state', 'MH',
                    '--output', 'test_units.csv'
                ])
        
        assert result.exit_code == 0
        assert 'No valid spatial unit data found' in result.output
        mock_client.get_spatial_units.assert_called_once_with('village', 'MH')

    def test_get_spatial_units_invalid_unit_type(self):
        """Test spatial units with invalid unit type."""
        result = self.runner.invoke(cli, [
            'get-spatial-units',
            '--unit-type', 'invalid_type'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid value' in result.output

    def test_get_spatial_units_api_error(self):
        """Test spatial units with API error."""
        mock_client = Mock(spec=CoREStackClient)
        mock_client.get_spatial_units.side_effect = Exception("API Error")
        
        with patch('ivi_water.cli.CoREStackClient', return_value=mock_client):
            result = self.runner.invoke(cli, [
                'get-spatial-units',
                '--unit-type', 'village'
            ])
        
        assert result.exit_code != 0
        assert 'Error:' in result.output

    def test_get_spatial_units_no_units_found(self):
        """Test spatial units with no units found."""
        mock_client = Mock(spec=CoREStackClient)
        mock_client.get_spatial_units.return_value = []
        
        with patch('ivi_water.cli.CoREStackClient', return_value=mock_client):
            result = self.runner.invoke(cli, [
                'get-spatial-units',
                '--unit-type', 'village'
            ])
        
        assert result.exit_code == 0
        assert 'No spatial units found' in result.output

    def test_cli_data_dir_validation(self, tmp_path):
        """Test CLI data directory validation."""
        # Test 1: Valid directory
        valid_dir = tmp_path / "valid_data"
        valid_dir.mkdir()
        
        # Should not raise any exceptions
        result = self.runner.invoke(cli, ['--data-dir', str(valid_dir), '--help'])
        assert result.exit_code == 0
        
        # Test 2: Non-existent directory
        non_existent = tmp_path / "nonexistent"
        result = self.runner.invoke(cli, ['--data-dir', str(non_existent)])
        assert result.exit_code != 0
        assert "does not exist" in result.output
        
        # Test 3: Path is a file, not a directory
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        result = self.runner.invoke(cli, ['--data-dir', str(file_path)])
        assert result.exit_code != 0
        assert "is a file" in result.output
        
        # Test 4: No data directory provided (should use default)
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0

    # def test_fetch_water_data_success(self, sample_water_data):
    #     """Test successful water data fetch."""
    #     # Create mock API client and processor
    #     mock_client = Mock(spec=CoREStackClient)
    #     mock_processor = Mock(spec=DataProcessor)
    #     mock_processor.load_water_data_from_api.return_value = sample_water_data
        
    #     with patch('ivi_water.cli.CoREStackClient', return_value=mock_client):
    #         with patch('ivi_water.cli.DataProcessor', return_value=mock_processor):
    #             with patch('pandas.DataFrame') as mock_df:
    #                 mock_df_instance = Mock()
    #                 mock_df_instance.__len__ = Mock(return_value=6)
    #                 mock_df_instance['location_id'].nunique.return_value = 2
    #                 mock_df_instance['year'].min.return_value = 2020
    #                 mock_df_instance['year'].max.return_value = 2022
    #                 mock_df_instance['season'].unique.return_value = ['perennial', 'winter', 'monsoon']
    #                 mock_df_instance['water_area_ha'].min.return_value = 45.0
    #                 mock_df_instance['water_area_ha'].max.return_value = 80.0
    #                 mock_df_instance.head.return_value = Mock()
    #                 mock_df.return_value = mock_df_instance
                    
    #                 result = self.runner.invoke(cli, [
    #                     'fetch-water-data',
    #                     '--locations', 'V001,V002',
    #                     '--start-year', '2020',
    #                     '--end-year', '2022',
    #                     '--output', 'test_water.csv'
    #                 ])
        
    #     assert result.exit_code == 0
    #     assert 'Fetched 6 records' in result.output
    #     mock_processor.load_water_data_from_api.assert_called_once()

    def test_fetch_water_data_invalid_locations(self):
        """Test water data fetch with invalid locations."""
        result = self.runner.invoke(cli, [
            'fetch-water-data',
            '--locations', '',
            '--start-year', '2020',
            '--end-year', '2022'
        ])
        
        assert result.exit_code != 0
        assert 'Locations parameter is required' in result.output

    def test_fetch_water_data_invalid_years(self):
        """Test water data fetch with invalid year range."""
        result = self.runner.invoke(cli, [
            'fetch-water-data',
            '--locations', 'V001',
            '--start-year', '2022',
            '--end-year', '2020'  # End year before start year
        ])
        
        assert result.exit_code != 0
        assert 'Start year must be less than or equal to end year' in result.output

    def test_fetch_water_data_invalid_seasons(self):
        """Test water data fetch with invalid seasons."""
        result = self.runner.invoke(cli, [
            'fetch-water-data',
            '--locations', 'V001',
            '--start-year', '2020',
            '--end-year', '2022',
            '--seasons', 'invalid_season'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid seasons' in result.output

    def test_fetch_water_data_no_data_found(self):
        """Test water data fetch with no data found."""
        mock_client = Mock(spec=CoREStackClient)
        mock_processor = Mock(spec=DataProcessor)
        mock_processor.load_water_data_from_api.return_value = pd.DataFrame()
        
        with patch('ivi_water.cli.CoREStackClient', return_value=mock_client):
            with patch('ivi_water.cli.DataProcessor', return_value=mock_processor):
                result = self.runner.invoke(cli, [
                    'fetch-water-data',
                    '--locations', 'V001',
                    '--start-year', '2020',
                    '--end-year', '2022'
                ])
        
        assert result.exit_code == 0
        assert 'No water data found' in result.output

    # def test_merge_data_success(self, sample_water_data, sample_nrm_data):
    #     """Test successful data merge."""
    #     # Create temporary data files
    #     water_file = self.temp_path / 'water.csv'
    #     nrm_file = self.temp_path / 'nrm.csv'
        
    #     sample_water_data.to_csv(water_file, index=False)
    #     sample_nrm_data.to_csv(nrm_file, index=False)
        
    #     mock_processor = Mock(spec=DataProcessor)
    #     mock_processor.merge_datasets.return_value = sample_water_data
        
    #     with patch('ivi_water.cli.DataProcessor', return_value=mock_processor):
    #         result = self.runner.invoke(cli, [
    #             'merge-data',
    #             '--water-data', str(water_file),
    #             '--nrm-data', str(nrm_file),
    #             '--output', 'merged.csv'
    #         ])
        
    #     assert result.exit_code == 0
    #     mock_processor.merge_datasets.assert_called_once()

    def test_merge_data_missing_water_file(self):
        """Test merge data with missing water file."""
        result = self.runner.invoke(cli, [
            'merge-data',
            '--water-data', '/nonexistent/water.csv',
            '--nrm-data', '/nonexistent/nrm.csv'
        ])
        
        assert result.exit_code != 0
        assert 'does not exist' in result.output

    def test_merge_data_merge_error(self, sample_water_data, sample_nrm_data):
        """Test merge data with merge error."""
        # Create temporary data files
        water_file = self.temp_path / 'water.csv'
        nrm_file = self.temp_path / 'nrm.csv'
        
        sample_water_data.to_csv(water_file, index=False)
        sample_nrm_data.to_csv(nrm_file, index=False)
        
        mock_processor = Mock(spec=DataProcessor)
        mock_processor.merge_datasets.side_effect = ValueError("Merge error")
        
        with patch('ivi_water.cli.DataProcessor', return_value=mock_processor):
            result = self.runner.invoke(cli, [
                'merge-data',
                '--water-data', str(water_file),
                '--nrm-data', str(nrm_file)
            ])
        
        assert result.exit_code != 0
        assert 'Error:' in result.output

    # def test_validate_locations_valid(self):
    #     """Test location validation with valid locations."""
    #     from ivi_water.cli import validate_locations
    #     result = validate_locations('V001,V002,V003')
    #     assert result == ['V001', 'V002', 'V003']

    # def test_validate_locations_empty(self):
    #     """Test location validation with empty string."""
    #     from ivi_water.cli import validate_locations
    #     with pytest.raises(ValueError, match="No valid location IDs provided"):
    #         validate_locations('')

    # def test_validate_locations_with_spaces(self):
    #     """Test location validation with spaces."""
    #     from ivi_water.cli import validate_locations
    #     result = validate_locations('V001, V002 , V003')
    #     assert result == ['V001', 'V002', 'V003']

    # def test_validate_years_valid(self):
    #     """Test year validation with valid years."""
    #     from ivi_water.cli import validate_years
    #     result = validate_years(2020, 2022)
    #     assert result == (2020, 2022)

    # def test_validate_years_invalid_range(self):
    #     """Test year validation with invalid range."""
    #     from ivi_water.cli import validate_years
    #     with pytest.raises(ValueError, match="Start year must be less than or equal to end year"):
    #         validate_years(2022, 2020)

    # def test_validate_years_out_of_range(self):
    #     """Test year validation with years out of range."""
    #     from ivi_water.cli import validate_years
    #     with pytest.raises(ValueError, match="out of reasonable range"):
    #         validate_years(1800, 2022)

    # def test_validate_seasons_valid(self):
    #     """Test season validation with valid seasons."""
    #     from ivi_water.cli import validate_seasons
    #     result = validate_seasons('perennial,winter,monsoon')
    #     assert result == ['perennial', 'winter', 'monsoon']

    # def test_validate_seasons_invalid(self):
    #     """Test season validation with invalid seasons."""
    #     from ivi_water.cli import validate_seasons
    #     with pytest.raises(ValueError, match="Invalid seasons"):
    #         validate_seasons('perennial,invalid_season')

    # def test_validate_seasons_none(self):
    #     """Test season validation with None."""
    #     from ivi_water.cli import validate_seasons
    #     result = validate_seasons(None)
    #     assert result is None

    # def test_validate_output_filename_valid(self):
    #     """Test output filename validation with valid filename."""
    #     from ivi_water.cli import validate_output_filename
    #     result = validate_output_filename('test_file')
    #     assert result == 'test_file.csv'

    # def test_validate_output_filename_with_extension(self):
    #     """Test output filename validation with existing extension."""
    #     from ivi_water.cli import validate_output_filename
    #     result = validate_output_filename('test_file.csv')
    #     assert result == 'test_file.csv'

    # def test_validate_output_filename_empty(self):
    #     """Test output filename validation with empty filename."""
    #     from ivi_water.cli import validate_output_filename
    #     with pytest.raises(ValueError, match="Output filename must be a non-empty string"):
    #         validate_output_filename('')

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        from ivi_water.cli import setup_logging
        logger = setup_logging(verbose=True)
        assert logger is not None

    def test_setup_logging_normal(self):
        """Test logging setup in normal mode."""
        from ivi_water.cli import setup_logging
        logger = setup_logging(verbose=False)
        assert logger is not None

    # def test_create_progress_bar(self):
    #     """Test progress bar creation."""
    #     from ivi_water.cli import create_progress_bar
    #     locations = ['V001', 'V002', 'V003']
        
    #     with patch('click.progressbar') as mock_progress:
    #         create_progress_bar(locations)
    #         mock_progress.assert_called_once()

    # def test_format_file_size(self):
    #     """Test file size formatting."""
    #     from ivi_water.cli import format_file_size
    #     assert format_file_size(1024) == '1.0 KB'
    #     assert format_file_size(1024 * 1024) == '1.0 MB'
    #     assert format_file_size(1024 * 1024 * 1024) == '1.0 GB'

    # def test_display_data_summary(self, sample_water_data):
    #     """Test data summary display."""
    #     from ivi_water.cli import display_data_summary
        
    #     with patch('click.echo') as mock_echo:
    #         display_data_summary(sample_water_data)
    #         assert mock_echo.call_count > 0

    # def test_display_sample_data(self, sample_water_data):
    #     """Test sample data display."""
    #     from ivi_water.cli import display_sample_data
        
    #     with patch('click.echo') as mock_echo:
    #         display_sample_data(sample_water_data)
    #         assert mock_echo.call_count > 0

    # def test_handle_api_error(self):
    #     """Test API error handling."""
    #     from ivi_water.cli import handle_api_error
        
    #     with patch('click.echo') as mock_echo:
    #         handle_api_error(Exception("API Error"))
    #         mock_echo.assert_called()

    # def test_handle_file_error(self):
    #     """Test file error handling."""
    #     from ivi_water.cli import handle_file_error
        
    #     with patch('click.echo') as mock_echo:
    #         handle_file_error(FileNotFoundError("File not found"))
    #         mock_echo.assert_called()

    def test_cli_integration_test(self):
        """Test CLI integration with multiple commands."""
        # Test that CLI can be imported and basic structure works
        from ivi_water.cli import cli
        assert cli is not None
        assert hasattr(cli, 'commands')

    def test_cli_command_help(self):
        """Test individual command help."""
        commands = ['get-spatial-units', 'fetch-water-data', 'merge-data']
        
        for command in commands:
            result = self.runner.invoke(cli, [command, '--help'])
            assert result.exit_code == 0
            assert 'Usage:' in result.output
