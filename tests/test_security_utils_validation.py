"""
Test Security Utils Validation

This module tests the validation functions in security_utils.py.
"""

import pytest
from ivi_water.security_utils import validate_safe_id, hash_data

class TestSecurityValidation:

    def test_validate_safe_id_valid(self):
        """Test validation with valid identifiers."""
        assert validate_safe_id("V001") == "V001"
        assert validate_safe_id("village-123") == "village-123"
        assert validate_safe_id("watershed_A") == "watershed_A"
        assert validate_safe_id("12345") == "12345"
        assert validate_safe_id("  V001  ") == "V001"  # Should strip whitespace

    def test_validate_safe_id_invalid(self):
        """Test validation with invalid identifiers."""
        # Special characters
        with pytest.raises(ValueError, match="Invalid identifier"):
            validate_safe_id("V001/../../etc")

        with pytest.raises(ValueError, match="Invalid identifier"):
            validate_safe_id("V001;")

        with pytest.raises(ValueError, match="Invalid identifier"):
            validate_safe_id("<script>")

        with pytest.raises(ValueError, match="Invalid identifier"):
            validate_safe_id("SELECT *")

        # Empty or non-string
        with pytest.raises(ValueError, match="Identifier cannot be empty"):
            validate_safe_id("")

        with pytest.raises(ValueError, match="Identifier cannot be empty"):
            validate_safe_id("   ")

        with pytest.raises(ValueError, match="Identifier must be a string"):
            validate_safe_id(None)

        with pytest.raises(ValueError, match="Identifier must be a string"):
            validate_safe_id(123)

    def test_hash_data(self):
        """Test SHA-256 hashing."""
        data = "test_data"
        expected_hash = "e7d87b738825c33824cf3fd32b7314161fc8c425129163ff5e7260fc7288da36"
        assert hash_data(data) == expected_hash

        # Verify it's consistent
        assert hash_data(data) == hash_data(data)

        # Verify different data produces different hash
        assert hash_data("other_data") != expected_hash
