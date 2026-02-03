"""
Test terminal sanitization functionality.
"""
import pytest
from ivi_water.security_utils import sanitize_for_terminal

def test_sanitize_for_terminal_basics():
    """Test basic string sanitization."""
    assert sanitize_for_terminal("Hello World") == "Hello World"
    assert sanitize_for_terminal("123") == "123"
    assert sanitize_for_terminal("") == ""

def test_sanitize_ansi_codes():
    """Test removal of ANSI escape sequences."""
    # Red text
    assert sanitize_for_terminal("\x1b[31mError\x1b[0m") == "Error"
    # Bold text
    assert sanitize_for_terminal("\x1b[1mBold\x1b[0m") == "Bold"
    # Complex sequence
    assert sanitize_for_terminal("\x1b[38;2;255;0;0mRGB\x1b[0m") == "RGB"

def test_sanitize_control_chars():
    """Test removal of control characters."""
    # Bell char
    assert sanitize_for_terminal("Ring\a") == "Ring"
    # Backspace (can hide characters)
    assert sanitize_for_terminal("Hide\bMe") == "HideMe"
    # Null char
    assert sanitize_for_terminal("Null\0") == "Null"

def test_sanitize_preserve_formatting():
    """Test escaping of allowed control characters."""
    # Newline
    assert sanitize_for_terminal("Line 1\nLine 2") == "Line 1\\nLine 2"
    # Tab
    assert sanitize_for_terminal("Col 1\tCol 2") == "Col 1\\tCol 2"
    # Carriage return
    assert sanitize_for_terminal("Line\rReturn") == "Line\\rReturn"

def test_sanitize_non_string():
    """Test handling of non-string inputs."""
    assert sanitize_for_terminal(123) == "123"
    assert sanitize_for_terminal(None) == "None"
    assert sanitize_for_terminal(1.5) == "1.5"
