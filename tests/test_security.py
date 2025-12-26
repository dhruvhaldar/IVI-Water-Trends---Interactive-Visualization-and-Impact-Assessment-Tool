
import os
import shutil
import pytest
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from ivi_water.data_processor import DataProcessor
from ivi_water.export_utils import sanitize_filename
from ivi_water.cli import cli

@pytest.fixture
def temp_data_dir():
    data_dir = Path('./test_data')
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    yield data_dir
    if data_dir.exists():
        shutil.rmtree(data_dir)

def test_sanitize_filename():
    """Test the filename sanitization logic."""
    assert sanitize_filename("normal_file.csv") == "normal_file.csv"
    assert sanitize_filename("../evil_file.csv") == "evil_file.csv"
    assert sanitize_filename("dir/file.csv") == "file.csv"
    assert sanitize_filename("file with spaces.csv") == "filewithspaces.csv"
    assert sanitize_filename("file$name.csv") == "filename.csv"

    with pytest.raises(ValueError):
        sanitize_filename("")

    with pytest.raises(ValueError):
        sanitize_filename("   ")

    with pytest.raises(ValueError):
        sanitize_filename("$$$")

def test_cli_path_traversal(temp_data_dir):
    """
    Test that CLI commands prevent path traversal in output filenames.
    """
    runner = CliRunner()
    output_dir = temp_data_dir / "outputs"
    output_dir.mkdir()

    # Mock the CoREStackClient
    with patch('ivi_water.cli.CoREStackClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_spatial_units.return_value = [{'id': 'V001', 'name': 'Test Village'}]

        # Try path traversal
        result = runner.invoke(cli, [
            '--output-dir', str(output_dir),
            'get-spatial-units',
            '--unit-type', 'village',
            '--output', '../evil_file'
        ])

        # Should succeed but write to sanitized path
        assert result.exit_code == 0

        # Check that file was NOT created outside
        outside_file = temp_data_dir / "evil_file.csv"
        assert not outside_file.exists(), "File should not exist outside output directory"

        # Check that file WAS created inside with sanitized name
        # sanitize_filename("../evil_file") -> "evil_file"
        # cli appends .csv if missing -> "evil_file.csv"
        expected_file = output_dir / "evil_file.csv"
        assert expected_file.exists(), f"File should exist at sanitized path: {expected_file}"

def test_export_processed_data_path_traversal(temp_data_dir):
    """
    Test that export_processed_data sanitizes filenames to prevent path traversal.
    """
    processor = DataProcessor(str(temp_data_dir))
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    # Attempt path traversal
    filename = "../outside_file"

    processor.export_processed_data(df, filename, format='csv')

    # Check that it did NOT write outside
    outside_file = temp_data_dir / "outside_file.csv"
    assert not outside_file.exists(), "File should not exist outside processed directory"

    # Check that it wrote to sanitized path
    # sanitize_filename("../outside_file") -> "outside_file"
    sanitized_filename = "outside_file.csv"
    processed_file = temp_data_dir / "processed" / sanitized_filename

    assert processed_file.exists(), f"File should exist at sanitized path: {processed_file}"

def test_export_processed_data_valid_filename(temp_data_dir):
    """
    Test that export_processed_data works correctly for valid filenames.
    """
    processor = DataProcessor(str(temp_data_dir))
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    filename = "valid_file"
    processor.export_processed_data(df, filename, format='csv')

    processed_file = temp_data_dir / "processed" / "valid_file.csv"
    assert processed_file.exists()

def test_export_processed_data_csv_injection(temp_data_dir):
    """
    Test that CSV injection is prevented.
    """
    processor = DataProcessor(str(temp_data_dir))
    # Malicious payload
    df = pd.DataFrame({'col1': ['=1+1', '@SUM(1,1)', '-1+1', '+1+1', 'Normal']})

    filename = "injection_test"
    processor.export_processed_data(df, filename, format='csv')

    processed_file = temp_data_dir / "processed" / "injection_test.csv"
    assert processed_file.exists()

    # Read back carefully (don't execute)
    # pandas read_csv reads it as is
    df_read = pd.read_csv(processed_file)

    # Check if ' was prepended
    assert df_read['col1'][0].startswith("'")
    assert df_read['col1'][1].startswith("'")
    assert df_read['col1'][2].startswith("'")
    assert df_read['col1'][3].startswith("'")
    assert not df_read['col1'][4].startswith("'")
