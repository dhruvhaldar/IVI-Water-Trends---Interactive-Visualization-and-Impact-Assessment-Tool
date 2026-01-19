
import os
import shutil
import pytest
import pandas as pd
import plotly.graph_objects as go
from ivi_water.visualizer import WaterTrendsVisualizer
from ivi_water.export_utils import ExportUtils
from ivi_water.security_utils import CSP_META_CONTENT

@pytest.fixture
def temp_output_dir(monkeypatch):
    output_dir = './test_csp_outputs'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    monkeypatch.setenv('OUTPUT_DIR', output_dir)
    yield output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'location_id': ['LOC1', 'LOC1', 'LOC2', 'LOC2'],
        'year': [2020, 2021, 2020, 2021],
        'season': ['winter', 'winter', 'winter', 'winter'],
        'water_area_ha': [10, 15, 20, 25],
        'water_body_count': [1, 1, 2, 2],
        'pond_presence': [0, 0, 1, 1]
    })

def verify_csp(filepath):
    assert os.path.exists(filepath), f"File not found: {filepath}"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'Content-Security-Policy' in content, f"CSP header missing in {filepath}"

    # Check for critical directives
    assert "default-src 'none'" in content
    assert "script-src 'unsafe-inline'" in content
    assert "style-src 'unsafe-inline'" in content
    assert "img-src 'self' data:" in content

def test_visualizer_save_figure_csp(temp_output_dir):
    """Test CSP in save_figure"""
    viz = WaterTrendsVisualizer()
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    viz.save_figure(fig, 'test_save_figure', 'html')

    filepath = os.path.join(temp_output_dir, 'test_save_figure.html')
    verify_csp(filepath)

def test_visualizer_dashboard_csp(temp_output_dir, sample_df):
    """Test CSP in create_multi_location_dashboard"""
    viz = WaterTrendsVisualizer()
    dash_path = os.path.join(temp_output_dir, 'dashboard.html')
    viz.create_multi_location_dashboard(sample_df, ['LOC1'], save_path=dash_path)

    verify_csp(dash_path)

def test_export_utils_visualization_csp(temp_output_dir):
    """Test CSP in create_visualization_exports"""
    exporter = ExportUtils(output_dir=temp_output_dir)
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    exporter.create_visualization_exports([fig], 'test_export', formats=['html'])

    filepath = os.path.join(temp_output_dir, 'test_export_1.html')
    verify_csp(filepath)

def test_export_utils_report_csp(temp_output_dir, sample_df):
    """Test CSP in generate_summary_report (HTML fallback)"""
    # Force REPORTLAB_AVAILABLE to False to trigger HTML generation if possible,
    # but since we can't easily mock module imports here without patching,
    # we'll call _create_html_report directly or hope reportlab isn't installed in test env.
    # Actually, ExportUtils prefers PDF if available.
    # Let's call _create_html_report directly to ensure coverage of that method.

    exporter = ExportUtils(output_dir=temp_output_dir)
    filepath = exporter._create_html_report(sample_df, exporter.output_dir, 'test_report')

    verify_csp(filepath)
