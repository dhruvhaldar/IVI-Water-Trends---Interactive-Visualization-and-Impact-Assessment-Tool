import pytest
from unittest.mock import MagicMock
from ivi_water.api_client import CoREStackClient, MAX_BATCH_SIZE as API_MAX_BATCH
from ivi_water.data_processor import DataProcessor, MAX_BATCH_SIZE as PROC_MAX_BATCH


def test_load_water_data_batch_limit():
    """Test that load_water_data_from_api enforces a batch size limit."""
    mock_client = MagicMock()
    processor = DataProcessor()

    # Create a large list of location IDs
    large_location_list = [f"loc_{i}" for i in range(PROC_MAX_BATCH + 1)]

    # This should raise a ValueError due to batch size limit
    mock_client.get_seasonal_water_data.return_value = {}

    with pytest.raises(ValueError, match=f"Batch size exceeds limit of {PROC_MAX_BATCH}"):
        processor.load_water_data_from_api(mock_client, large_location_list, 2020, 2021)


def test_load_water_data_within_limit():
    """Test that load_water_data_from_api accepts list within limit."""
    mock_client = MagicMock()
    processor = DataProcessor()

    # Create a list within limit
    valid_location_list = [f"loc_{i}" for i in range(PROC_MAX_BATCH)]

    # Mock return value to prevent failure on processing
    mock_client.get_seasonal_water_data.return_value = {
        "timeseries": [{"year": 2020, "seasons": {"monsoon": {"area_ha": 10}}}]
    }

    # Should not raise ValueError about batch limit
    # (Might raise other errors if mock isn't perfect, but we check message)
    try:
        processor.load_water_data_from_api(mock_client, valid_location_list, 2020, 2021)
    except ValueError as e:
        assert "Batch size exceeds limit" not in str(e)


def test_api_summary_batch_limit():
    """Test that get_water_trends_summary enforces a batch size limit."""
    client = CoREStackClient(api_key="test", base_url="https://example.com")

    # Mock _make_request to avoid actual network calls
    client._make_request = MagicMock(return_value={})

    large_location_list = [f"loc_{i}" for i in range(API_MAX_BATCH + 1)]

    with pytest.raises(ValueError, match=f"Batch size exceeds limit of {API_MAX_BATCH}"):
        client.get_water_trends_summary(large_location_list, 2020, 2021)


def test_api_summary_within_limit():
    """Test that get_water_trends_summary accepts list within limit."""
    client = CoREStackClient(api_key="test", base_url="https://example.com")
    client._make_request = MagicMock(return_value={})

    valid_location_list = [f"loc_{i}" for i in range(API_MAX_BATCH)]

    # Should not raise
    client.get_water_trends_summary(valid_location_list, 2020, 2021)
