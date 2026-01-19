import os
import shutil
import pytest
import plotly.graph_objects as go
from ivi_water.visualizer import WaterTrendsVisualizer


@pytest.fixture
def temp_output_dir():
    output_dir = "./test_viz_outputs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.environ["OUTPUT_DIR"] = output_dir
    yield output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def test_save_figure_path_traversal(temp_output_dir):
    """
    Test that save_figure sanitizes filenames to prevent path traversal.
    """
    viz = WaterTrendsVisualizer()
    fig = go.Figure()

    # Attempt path traversal
    payload = "../outside_file"

    # This should write to output_dir/outside_file.html, NOT output_dir/../outside_file.html
    viz.save_figure(fig, payload, "html")

    # Check if it wrote to the sanitized location
    sanitized_file = os.path.join(temp_output_dir, "outside_file.html")
    assert os.path.exists(
        sanitized_file
    ), f"File should exist at sanitized path: {sanitized_file}"

    # Verify it didn't write outside (relative to output_dir)
    # output_dir is ./test_viz_outputs
    # ../outside_file.html would be ./outside_file.html
    outside_file = "./outside_file.html"
    if os.path.exists(outside_file):
        # Clean up if it did (though it shouldn't if sanitized)
        os.remove(outside_file)
        pytest.fail("File was written outside output directory!")


def test_save_figure_valid_filename(temp_output_dir):
    """
    Test that save_figure works for valid filenames.
    """
    viz = WaterTrendsVisualizer()
    fig = go.Figure()

    filename = "test_chart"
    viz.save_figure(fig, filename, "html")

    expected_file = os.path.join(temp_output_dir, "test_chart.html")
    assert os.path.exists(expected_file)


def test_save_figure_csp_injection(temp_output_dir):
    """
    Test that save_figure injects Content Security Policy meta tag.
    """
    viz = WaterTrendsVisualizer()
    fig = go.Figure()

    filename = "test_csp"
    viz.save_figure(fig, filename, "html")

    filepath = os.path.join(temp_output_dir, f"{filename}.html")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Content-Security-Policy" in content
    assert "default-src 'none'" in content
    assert "script-src 'unsafe-inline'" in content
