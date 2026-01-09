
import pytest
from ivi_water.security_utils import validate_safe_id, MAX_ID_LENGTH
from ivi_water.export_utils import sanitize_filename, MAX_FILENAME_LENGTH

def test_validate_safe_id_length():
    """Test that validate_safe_id rejects identifiers that are too long."""
    # Test with valid length
    valid_id = "a" * MAX_ID_LENGTH
    assert validate_safe_id(valid_id) == valid_id

    # Test with invalid length
    invalid_id = "a" * (MAX_ID_LENGTH + 1)
    with pytest.raises(ValueError, match=f"Identifier exceeds maximum length of {MAX_ID_LENGTH}"):
        validate_safe_id(invalid_id)

def test_sanitize_filename_length():
    """Test that sanitize_filename rejects filenames that are too long."""
    # Test with valid length
    valid_filename = "a" * MAX_FILENAME_LENGTH
    assert sanitize_filename(valid_filename) == valid_filename

    # Test with invalid length
    invalid_filename = "a" * (MAX_FILENAME_LENGTH + 1)
    with pytest.raises(ValueError, match=f"Filename exceeds maximum length of {MAX_FILENAME_LENGTH}"):
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
