
import os
import shutil
import pytest
import pandas as pd
from pathlib import Path
from ivi_water.data_processor import DataProcessor

@pytest.fixture
def temp_data_dir():
    data_dir = Path('./test_data')
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    yield data_dir
    if data_dir.exists():
        shutil.rmtree(data_dir)

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
    # ../outside_file -> outside_file (because of os.path.basename)
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
