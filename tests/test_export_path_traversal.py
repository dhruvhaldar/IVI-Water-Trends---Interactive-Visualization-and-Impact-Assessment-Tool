
import os
import shutil
import pytest
import plotly.graph_objects as go
from pathlib import Path
from ivi_water.export_utils import ExportUtils, sanitize_filename

@pytest.fixture
def temp_output_dir(monkeypatch):
    output_dir = './test_traversal_outputs'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    monkeypatch.setenv('OUTPUT_DIR', output_dir)
    yield output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_create_visualization_exports_traversal(temp_output_dir):
    """Test that create_visualization_exports prevents path traversal"""
    exporter = ExportUtils(output_dir=temp_output_dir)
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

    # Prefix with traversal
    prefix = '../traversal_test'

    # This should NOT raise exception, but should sanitize the filename
    # resulting in 'traversal_test_1.html' inside temp_output_dir
    exported_files = exporter.create_visualization_exports([fig], prefix, formats=['html'])

    # Check that file was created in the correct directory
    expected_filename = 'traversal_test_1.html'
    expected_path = os.path.join(temp_output_dir, expected_filename)

    assert os.path.exists(expected_path), "Sanitized file was not created in output directory"
    assert exported_files[0] == str(Path(expected_path).absolute())

    # Check that file was NOT created outside
    traversal_path = os.path.join(temp_output_dir, '../traversal_test_1.html')
    assert not os.path.exists(traversal_path), "File created outside output directory!"

def test_sanitize_filename_traversal():
    """Unit test for sanitize_filename with traversal characters"""
    assert sanitize_filename('../test') == 'test'

    # On Linux, \ is just a character, replaced by _
    # On Windows, it is a separator.
    # Since we are likely on Linux, assume Linux behavior for now or handle both.
    if os.sep == '/':
        assert sanitize_filename('..\\test') == '_test' # .. becomes .. (leading dots removed), \ becomes _
        # Wait, ..\test -> .._test -> _test. Correct.
    else:
        assert sanitize_filename('..\\test') == 'test'

    assert sanitize_filename('/tmp/test') == 'test'
    assert sanitize_filename('./test') == 'test'
    assert sanitize_filename('dir/../test') == 'test'
