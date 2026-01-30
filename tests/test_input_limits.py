import pytest
from unittest.mock import MagicMock, patch
from ivi_water.security_utils import validate_safe_id, MAX_ID_LENGTH
from ivi_water.export_utils import sanitize_filename, MAX_FILENAME_LENGTH
from ivi_water.api_client import CoREStackClient, MAX_BATCH_SIZE as API_MAX_BATCH_SIZE
from ivi_water.data_processor import DataProcessor, MAX_BATCH_SIZE as DP_MAX_BATCH_SIZE


def test_validate_safe_id_length():
    """Test that validate_safe_id rejects identifiers that are too long."""
    # Test with valid length
    valid_id = "a" * MAX_ID_LENGTH
    assert validate_safe_id(valid_id) == valid_id

    # Test with invalid length
    invalid_id = "a" * (MAX_ID_LENGTH + 1)
    with pytest.raises(
        ValueError, match=f"Identifier exceeds maximum length of {MAX_ID_LENGTH}"
    ):
        validate_safe_id(invalid_id)


def test_sanitize_filename_length():
    """Test that sanitize_filename rejects filenames that are too long."""
    # Test with valid length
    valid_filename = "a" * MAX_FILENAME_LENGTH
    assert sanitize_filename(valid_filename) == valid_filename

    # Test with invalid length
    invalid_filename = "a" * (MAX_FILENAME_LENGTH + 1)
    with pytest.raises(
        ValueError, match=f"Filename exceeds maximum length of {MAX_FILENAME_LENGTH}"
    ):
        sanitize_filename(invalid_filename)


def test_validate_safe_id_validity():
    """Test basic validity checks for safe_id."""
    assert validate_safe_id("valid-id_123") == "valid-id_123"
    with pytest.raises(ValueError, match="Invalid identifier"):
        validate_safe_id("invalid id")
    with pytest.raises(ValueError, match="Invalid identifier"):
        validate_safe_id("invalid/id")


def test_sanitize_filename_validity():
    """Test basic validity checks for sanitize_filename."""
    assert sanitize_filename("valid_filename.csv") == "valid_filename.csv"
    assert sanitize_filename("path/to/file.csv") == "file.csv"

    with pytest.raises(ValueError, match="Filename contains no valid characters"):
        sanitize_filename("..")


def test_batch_size_limits():
    """Test that batch operations enforce MAX_BATCH_SIZE."""
    # Ensure constants match (sanity check)
    assert API_MAX_BATCH_SIZE == 100
    assert DP_MAX_BATCH_SIZE == 100

    # Create a list larger than the limit
    large_batch = [f"loc_{i}" for i in range(API_MAX_BATCH_SIZE + 1)]
    valid_batch = [f"loc_{i}" for i in range(API_MAX_BATCH_SIZE)]

    # Test CoREStackClient
    # Mock environment variables to allow instantiation without real API key
    with patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
        client = CoREStackClient()

        # Should raise ValueError
        with pytest.raises(
            ValueError, match=f"Batch size {len(large_batch)} exceeds maximum limit"
        ):
            client.get_water_trends_summary(large_batch, 2020, 2021)

        # Should NOT raise ValueError (mock request will fail later but validation passes)
        # We mock _make_request to avoid actual network call failure
        client._make_request = MagicMock(return_value={"data": {}})
        client.get_water_trends_summary(valid_batch, 2020, 2021)

    # Test DataProcessor
    processor = DataProcessor()
    mock_client = MagicMock()
    mock_client.get_seasonal_water_data.return_value = {}  # Mock return for valid call

    # Should raise ValueError
    with pytest.raises(
        ValueError, match=f"Batch size {len(large_batch)} exceeds maximum limit"
    ):
        processor.load_water_data_from_api(mock_client, large_batch, 2020, 2021)

    # Should NOT raise ValueError (will run but return empty/mocked data)
    # This verifies that valid batch size is accepted
    try:
        processor.load_water_data_from_api(mock_client, valid_batch, 2020, 2021)
    except ValueError as e:
        # It might raise ValueError because no data loaded (empty mock response),
        # but NOT "Batch size ... exceeds"
        assert "Batch size" not in str(e)
