
import pytest
from ivi_water.security_utils import validate_safe_id

def test_validate_safe_id_valid():
    """Test valid identifiers."""
    validate_safe_id("V001")
    validate_safe_id("my-location")
    validate_safe_id("loc_123")
    validate_safe_id("loc.1")
    validate_safe_id("A.B-C_1")

def test_validate_safe_id_invalid_chars():
    """Test identifiers with invalid characters."""
    with pytest.raises(ValueError, match="contains invalid characters"):
        validate_safe_id("loc<script>")

    with pytest.raises(ValueError, match="contains invalid characters"):
        validate_safe_id("loc space")

    with pytest.raises(ValueError, match="contains invalid characters"):
        validate_safe_id("loc$")

def test_validate_safe_id_path_traversal():
    """Test identifiers with path traversal sequences."""
    with pytest.raises(ValueError, match="contains invalid path characters"):
        validate_safe_id("../etc/passwd")

    with pytest.raises(ValueError, match="contains invalid path characters"):
        validate_safe_id("loc/1")

    with pytest.raises(ValueError, match="contains invalid path characters"):
        validate_safe_id("loc\\1")

    with pytest.raises(ValueError, match="contains invalid path characters"):
        validate_safe_id("..")

def test_validate_safe_id_empty():
    """Test empty identifiers."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_safe_id("")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_safe_id("   ")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_safe_id(None) # type: ignore
