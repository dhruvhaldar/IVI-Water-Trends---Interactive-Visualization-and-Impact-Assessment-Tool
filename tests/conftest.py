"""
Pytest configuration and shared fixtures for IVI Water Trends tests.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Import modules to test
from ivi_water.data_processor import DataProcessor
from ivi_water.api_client import CoREStackClient
from ivi_water.visualizer import WaterTrendsVisualizer
from ivi_water.export_utils import ExportUtils


@pytest.fixture
def sample_water_data():
    """Create sample water data DataFrame for testing."""
    return pd.DataFrame({
        'location_id': ['V001', 'V001', 'V001', 'V002', 'V002', 'V002'],
        'year': [2020, 2021, 2022, 2020, 2021, 2022],
        'season': ['perennial', 'winter', 'monsoon', 'perennial', 'winter', 'monsoon'],
        'water_area_ha': [50.0, 45.0, 60.0, 75.0, 70.0, 80.0],
        'water_body_count': [5, 4, 6, 8, 7, 9],
        'data_quality_score': [0.9, 0.85, 0.95, 0.92, 0.88, 0.94]
    })


@pytest.fixture
def sample_nrm_data():
    """Create sample NRM data DataFrame for testing."""
    return pd.DataFrame({
        'location_id': ['V001', 'V002'],
        'year': [2021, 2021],
        'pond_presence': [1, 0],
        'pond_area_ha': [2.5, 0.0],
        'watershed_area_ha': [100.0, 120.0]
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_api_response():
    """Mock API response data."""
    return {
        'data': [
            {'id': 'V001', 'name': 'Village 001', 'state': 'MH'},
            {'id': 'V002', 'name': 'Village 002', 'state': 'MH'}
        ]
    }


@pytest.fixture
def data_processor(temp_dir):
    """Create DataProcessor instance with temporary directory."""
    return DataProcessor(str(temp_dir))


@pytest.fixture
def visualizer():
    """Create WaterTrendsVisualizer instance for testing."""
    return WaterTrendsVisualizer(theme='plotly_white', height=600, width=1000)


@pytest.fixture
def export_utils(temp_dir):
    """Create ExportUtils instance with temporary directory."""

    print("Export Utils",str(temp_dir))
    return ExportUtils(str(temp_dir))


@pytest.fixture
def mock_api_client():
    """Create mock CoREStackClient for testing."""
    client = Mock(spec=CoREStackClient)
    client.get_spatial_units.return_value = [
        {'id': 'V001', 'name': 'Village 001', 'state': 'MH'},
        {'id': 'V002', 'name': 'Village 002', 'state': 'MH'}
    ]
    client.get_seasonal_water_data.return_value = {
        'timeseries': [
            {'year': 2020, 'season': 'perennial', 'water_area_ha': 50.0},
            {'year': 2021, 'season': 'winter', 'water_area_ha': 45.0}
        ]
    }
    return client


@pytest.fixture
def sample_merged_data(sample_water_data, sample_nrm_data):
    """Create sample merged water and NRM data."""
    # Create merged data with some matching records
    merged = sample_water_data.copy()
    merged['pond_presence'] = [1, 1, 1, 0, 0, 0]  # Add intervention data
    merged['pond_area_ha'] = [2.5, 2.5, 2.5, 0.0, 0.0, 0.0]
    merged['watershed_area_ha'] = [100.0, 100.0, 100.0, 120.0, 120.0, 120.0]
    merged['crop_yield_ton_per_ha'] = [2.5, 2.7, 2.8, 2.0, 2.1, 2.2]  # Add crop yield data
    return merged


# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv('DATA_DIR', './test_data')
    monkeypatch.setenv('OUTPUT_DIR', './test_outputs')
    monkeypatch.setenv('DEFAULT_CHART_THEME', 'plotly_white')
    monkeypatch.setenv('CHART_HEIGHT', '600')
    monkeypatch.setenv('CHART_WIDTH', '1000')
    monkeypatch.setenv('EXPORT_DPI', '300')


# Mock logging during tests to reduce noise
@pytest.fixture(autouse=True)
def disable_logging(monkeypatch):
    """Disable logging during tests."""
    import logging
    mock_logger = Mock()
    monkeypatch.setattr(logging, 'getLogger', lambda name='': mock_logger)
