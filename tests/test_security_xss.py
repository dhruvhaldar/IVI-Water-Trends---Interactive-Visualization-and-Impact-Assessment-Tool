
import pytest
import pandas as pd
import os
import html
from pathlib import Path
from ivi_water.export_utils import ExportUtils

def test_html_report_xss_prevention(tmp_path):
    """
    Test that HTML report generation escapes malicious column names
    to prevent XSS vulnerabilities.
    """
    # Create a DataFrame with a malicious column name
    malicious_column = "<script>alert('XSS')</script>"
    df = pd.DataFrame({
        'location_id': ['V001', 'V002'],
        'year': [2020, 2021],
        malicious_column: [1, 2]
    })

    exporter = ExportUtils(output_dir=str(tmp_path))

    # Generate HTML report directly
    filename = "xss_test"
    filepath = exporter._create_html_report(df, tmp_path, filename)

    assert os.path.exists(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    # Content should NOT contain the raw script tag
    assert malicious_column not in content, "XSS Vulnerability detected: Raw script tag found in output"

    # Content SHOULD contain the escaped version
    escaped_column = html.escape(malicious_column)
    assert escaped_column in content, "Escaped column name not found in output"

def test_sanitize_dataframe_csv_injection():
    """
    Test that sanitize_dataframe correctly escapes CSV injection characters.
    """
    df = pd.DataFrame({
        'safe': ['safe', 'value'],
        'unsafe_equals': ['=cmd| /C calc!A0', 'normal'],
        'unsafe_plus': ['+cmd| /C calc!A0', 'normal'],
        'unsafe_minus': ['-cmd| /C calc!A0', 'normal'],
        'unsafe_at': ['@cmd| /C calc!A0', 'normal']
    })

    sanitized_df = pd.DataFrame({
        'safe': ['safe', 'value'],
        'unsafe_equals': ["'=cmd| /C calc!A0", 'normal'],
        'unsafe_plus': ["'+cmd| /C calc!A0", 'normal'],
        'unsafe_minus': ["'-cmd| /C calc!A0", 'normal'],
        'unsafe_at': ["'@cmd| /C calc!A0", 'normal']
    })

    # Reuse the existing function from export_utils
    from ivi_water.export_utils import sanitize_dataframe

    result = sanitize_dataframe(df)

    pd.testing.assert_frame_equal(result, sanitized_df)
