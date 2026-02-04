
import pytest
from ivi_water.export_utils import sanitize_filename

class TestReservedFilenames:
    """Test suite for Windows reserved filename protection"""

    def test_reserved_names_basic(self):
        """Test basic reserved names are sanitized"""
        reserved_list = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM9", "LPT1", "LPT9"
        ]

        for name in reserved_list:
            sanitized = sanitize_filename(name)
            assert sanitized == f"_{name}", f"Failed to sanitize {name}"

    def test_reserved_names_case_insensitive(self):
        """Test case insensitivity for reserved names"""
        reserved_list = [
            "con", "PrN", "aux", "nul",
            "com1", "lpt9"
        ]

        for name in reserved_list:
            sanitized = sanitize_filename(name)
            # sanitize_filename preserves case of input but checks against upper
            assert sanitized == f"_{name}", f"Failed to sanitize {name}"

    def test_reserved_names_with_extensions(self):
        """Test reserved names with extensions are sanitized"""
        cases = [
            ("CON.txt", "_CON.txt"),
            ("aux.json", "_aux.json"),
            ("NUL.tar.gz", "_NUL.tar.gz"),
            ("com1.csv", "_com1.csv"),
            ("lpt5.xml", "_lpt5.xml")
        ]

        for input_name, expected in cases:
            assert sanitize_filename(input_name) == expected

    def test_safe_names(self):
        """Test safe names are not affected"""
        safe_names = [
            "console.txt",
            "auxiliary.json",
            "null.tar.gz",
            "computer.jpg",
            "com10.txt",  # COM10 is not reserved (only 1-9)
            "lpt0.txt",   # LPT0 is not reserved
            "my_con.txt"
        ]

        for name in safe_names:
            assert sanitize_filename(name) == name

    def test_reserved_name_as_part_of_filename(self):
        """Test reserved names appearing as part of filename are safe if not exact match of base"""
        # "con_file.txt" -> base "con_file" -> Safe
        assert sanitize_filename("con_file.txt") == "con_file.txt"
        # "file_con.txt" -> base "file_con" -> Safe
        assert sanitize_filename("file_con.txt") == "file_con.txt"

    def test_sanitization_order(self):
        """Test that reserved check happens after character sanitization"""
        # "CON/AUX.txt" -> path traversal removed -> "AUX.txt" -> Reserved -> "_AUX.txt"
        # Wait, os.path.basename("CON/AUX.txt") -> "AUX.txt".
        assert sanitize_filename("CON/AUX.txt") == "_AUX.txt"

        # "CON\AUX.txt" -> on Linux, \ replaced by _ -> "CON_AUX.txt" -> Safe
        # on Windows, \ is separator -> "AUX.txt" -> "_AUX.txt"
        # Since we are likely on Linux, let's assume replacement behavior
        import os
        if os.sep == '/':
            assert sanitize_filename("CON\\AUX.txt") == "CON_AUX.txt"
