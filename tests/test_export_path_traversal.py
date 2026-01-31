
import pytest
import pandas as pd
from ivi_water.export_utils import ExportUtils


# Mock data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'location_id': ['V001', 'V002'],
        'year': [2020, 2021],
        'water_area_ha': [10.5, 20.0]
    })


def test_create_visualization_exports_path_traversal(tmp_path, sample_df):
    """Test that create_visualization_exports sanitizes filename prefix."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    exporter = ExportUtils(str(output_dir))

    # Create a dummy figure (mocking Plotly figure)
    class MockFigure:
        def to_html(self):
            return "<html></html>"

        def write_image(self, path, **kwargs):
            # Create empty file
            with open(path, 'w') as f:
                f.write("image data")

    figures = [MockFigure()]

    # Try path traversal
    traversal_filename = "../hacked"

    # This should write to output_dir / "hacked_1.html" instead of parent dir
    exporter.create_visualization_exports(
        figures, traversal_filename, formats=["html"]
    )

    # Assertions
    # 1. Check that file DOES NOT exist in parent directory
    hacked_file = output_dir.parent / "hacked_1.html"
    assert not hacked_file.exists(), (
        f"Path traversal succeeded! File written to {hacked_file}"
    )

    # 2. Check that file DOES exist in output directory (sanitized)
    # The sanitize_filename("..") removes directory separators and special chars
    # sanitize_filename("../hacked") -> "hacked"
    sanitized_file = output_dir / "hacked_1.html"
    assert sanitized_file.exists(), (
        f"Sanitized file not found at {sanitized_file}"
    )


def test_generate_summary_report_path_traversal(tmp_path, sample_df):
    """Test that generate_summary_report sanitizes filename."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    exporter = ExportUtils(str(output_dir))

    traversal_filename = "../hacked_report"

    # This uses internal _create_html_report or _create_pdf_report
    path = exporter.generate_summary_report(
        sample_df, filename=traversal_filename
    )

    # Assertions
    # Check that returned path is correct
    assert str(output_dir) in path

    # Check that file DOES NOT exist in parent directory
    # Depending on environment, it generates PDF or HTML
    is_pdf = path.endswith(".pdf")
    ext = ".pdf" if is_pdf else ".html"

    hacked_file = output_dir.parent / f"hacked_report{ext}"
    assert not hacked_file.exists(), (
        f"Path traversal succeeded! File written to {hacked_file}"
    )

    # Check that file DOES exist in output directory
    sanitized_file = output_dir / f"hacked_report{ext}"
    assert sanitized_file.exists(), (
        f"Sanitized file not found at {sanitized_file}"
    )


def test_generate_short_summary_path_traversal(tmp_path, sample_df):
    """Test that generate_short_summary sanitizes filename."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    exporter = ExportUtils(str(output_dir))

    traversal_filename = "../hacked_short"

    exporter.generate_short_summary(
        sample_df, filename=traversal_filename
    )

    hacked_file = output_dir.parent / "hacked_short.txt"
    assert not hacked_file.exists(), (
        f"Path traversal succeeded! File written to {hacked_file}"
    )

    sanitized_file = output_dir / "hacked_short.txt"
    assert sanitized_file.exists(), (
        f"Sanitized file not found at {sanitized_file}"
    )


def test_generate_detailed_report_path_traversal(tmp_path, sample_df):
    """Test that generate_detailed_report sanitizes filename."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    exporter = ExportUtils(str(output_dir))

    traversal_filename = "../hacked_detailed"

    exporter.generate_detailed_report(
        sample_df, filename=traversal_filename
    )

    # Detailed report generates multiple files.
    # We check the index file which is returned.

    hacked_file = output_dir.parent / "hacked_detailed_index.txt"
    assert not hacked_file.exists(), (
        f"Path traversal succeeded! File written to {hacked_file}"
    )

    sanitized_file = output_dir / "hacked_detailed_index.txt"
    assert sanitized_file.exists(), (
        f"Sanitized file not found at {sanitized_file}"
    )
