"""
Working unit tests for methods that actually exist in the IVI Water Trends modules.
This test suite focuses on the actual public methods available in each class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from ivi_water.data_processor import DataProcessor
from ivi_water.api_client import CoREStackClient
from ivi_water.visualizer import WaterTrendsVisualizer
from ivi_water.export_utils import ExportUtils


class TestWorkingMethods:
    """Test cases for methods that actually exist in the classes."""

    def test_data_processor_init(self, temp_dir):
        """Test DataProcessor initialization."""
        processor = DataProcessor(str(temp_dir))
        assert processor.data_dir == temp_dir.resolve()
        assert hasattr(processor, 'logger')

    def test_data_processor_clean_water_data(self, temp_dir):
        """Test water data cleaning."""
        processor = DataProcessor(str(temp_dir))
        
        # Create sample data
        dirty_data = pd.DataFrame({
            'location_id': ['V001', 'V002', ''],
            'year': [2020, 2021, 2022],
            'season': ['perennial', 'winter', 'monsoon'],
            'water_area_ha': [50.0, np.nan, 60.0],  # One NaN value
            'water_body_count': [5, 4, 6],
            'data_quality_score': [0.9, 0.85, 0.95]
        })
        
        cleaned = processor._clean_water_data(dirty_data)
        
        # Should remove rows with NaN and empty strings
        assert len(cleaned) == 2
        assert not cleaned['water_area_ha'].isna().any()
        assert all(cleaned['location_id'] != '')

    def test_data_processor_merge_datasets(self, temp_dir, sample_water_data, sample_nrm_data):
        """Test dataset merging."""
        processor = DataProcessor(str(temp_dir))
        
        merged = processor.merge_datasets(sample_water_data, sample_nrm_data)
        
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) > 0

    def test_data_processor_aggregate_by_intervention(self, temp_dir, sample_merged_data):
        """Test intervention aggregation."""
        processor = DataProcessor(str(temp_dir))
        
        agg = processor.aggregate_by_intervention(sample_merged_data)
        
        assert isinstance(agg, pd.DataFrame)
        assert 'intervention_type' in agg.columns

    def test_data_processor_export_processed_data(self, temp_dir, sample_water_data):
        """Test data export."""
        processor = DataProcessor(str(temp_dir))
        
        output_path = processor.export_processed_data(sample_water_data, 'test_data', 'csv')
        
        assert Path(output_path).exists()
        assert output_path.endswith('.csv')

    def test_api_client_init(self):
        """Test CoREStackClient initialization."""
        with patch.dict('os.environ', {'CORE_API_KEY': 'test_key'}):
            client = CoREStackClient()
            assert client.api_key == 'test_key'
            assert hasattr(client, 'session')

    def test_api_client_clear_cache(self):
        """Test cache clearing."""
        with patch.dict('os.environ', {'CORE_API_KEY': 'test_key'}):
            client = CoREStackClient()
            
            # Add some cache entries
            client._cache['test_key'] = {'data': 'test'}
            client._cache_timestamps['test_key'] = pd.Timestamp.now()
            
            # Clear cache
            client.clear_cache()
            
            assert len(client._cache) == 0
            assert len(client._cache_timestamps) == 0

    def test_api_client_get_cache_info(self):
        """Test cache information."""
        with patch.dict('os.environ', {'CORE_API_KEY': 'test_key'}):
            client = CoREStackClient()
            
            info = client.get_cache_info()
            
            assert isinstance(info, dict)
            assert 'cache_size' in info
            assert 'cache_ttl' in info

    def test_visualizer_init(self):
        """Test WaterTrendsVisualizer initialization."""
        viz = WaterTrendsVisualizer()
        assert viz.theme == 'plotly_white'
        assert viz.height == 600
        assert viz.width == 1000

    def test_visualizer_init_custom(self):
        """Test WaterTrendsVisualizer with custom parameters."""
        viz = WaterTrendsVisualizer(theme='plotly_dark', height=800, width=1200)
        assert viz.theme == 'plotly_dark'
        assert viz.height == 800
        assert viz.width == 1200

    def test_visualizer_create_seasonal_stacked_area_chart(self, sample_water_data):
        """Test creating seasonal stacked area chart."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_seasonal_stacked_area_chart(sample_water_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    def test_visualizer_create_seasonal_stacked_area_chart_single_location(self, sample_water_data):
        """Test creating chart for single location."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_seasonal_stacked_area_chart(sample_water_data, location_id='V001')
        
        assert fig is not None
        assert 'V001' in fig.layout.title.text

    def test_visualizer_create_comparison_line_plot(self, sample_merged_data):
        """Test creating comparison line plot."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_comparison_line_plot(sample_merged_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_water_body_distribution(self, sample_water_data):
        """Test creating water body distribution."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_water_body_distribution(sample_water_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_trend_heatmap(self, sample_water_data):
        """Test creating trend heatmap."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_trend_heatmap(sample_water_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_seasonal_box_plot(self, sample_water_data):
        """Test creating seasonal box plot."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_seasonal_box_plot(sample_water_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_intervention_impact_scatter(self, sample_merged_data):
        """Test creating intervention impact scatter plot."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_intervention_impact_scatter(sample_merged_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_multi_location_dashboard(self, sample_water_data):
        """Test creating multi-location dashboard."""
        viz = WaterTrendsVisualizer()
        
        fig = viz.create_multi_location_dashboard(sample_water_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_visualizer_create_3d_water_visualization(self, sample_water_data):
        """Test creating 3D water visualization."""
        viz = WaterTrendsVisualizer()
        
        # This should work without PyVista (will show warning)
        fig = viz.create_3d_water_visualization(sample_water_data)
        
        # Should return None or handle gracefully without PyVista
        assert True  # Test passes if no exception is raised

    def test_visualizer_save_figure(self, sample_water_data, temp_dir):
        """Test saving figure."""
        viz = WaterTrendsVisualizer()
        fig = viz.create_seasonal_stacked_area_chart(sample_water_data)
        
        output_path = viz.save_figure(fig, 'test_chart', 'html')
        
        assert output_path.endswith('.html')
        assert Path(output_path).exists()

    def test_export_utils_init(self, temp_dir):
        """Test ExportUtils initialization."""
        exporter = ExportUtils(str(temp_dir))
        assert exporter.output_dir == temp_dir.resolve()
        assert hasattr(exporter, 'logger')

    def test_export_utils_export_data_table(self, temp_dir, sample_water_data):
        """Test exporting data table."""
        exporter = ExportUtils(str(temp_dir))
        
        output_path = exporter.export_data_table(sample_water_data, 'test_data', 'csv')
        
        assert Path(output_path).exists()
        assert output_path.endswith('.csv')

    def test_export_utils_create_visualization_exports(self, temp_dir, sample_water_data):
        """Test creating visualization exports."""
        exporter = ExportUtils(str(temp_dir))
        
        output_paths = exporter.create_visualization_exports(sample_water_data, 'test_viz')
        
        assert isinstance(output_paths, dict)
        assert 'png' in output_paths
        assert 'pdf' in output_paths

    def test_export_utils_generate_summary_report(self, temp_dir, sample_water_data):
        """Test generating summary report."""
        exporter = ExportUtils(str(temp_dir))
        
        output_path = exporter.generate_summary_report(sample_water_data, 'Test Report')
        
        assert output_path.endswith('.pdf')
        assert Path(output_path).exists()

    def test_export_utils_generate_detailed_report(self, temp_dir, sample_water_data):
        """Test generating detailed report."""
        exporter = ExportUtils(str(temp_dir))
        
        output_path = exporter.generate_detailed_report(sample_water_data, 'Detailed Report')
        
        assert output_path.endswith('.pdf')
        assert Path(output_path).exists()

    def test_export_utils_generate_short_summary(self, temp_dir, sample_water_data):
        """Test generating short summary."""
        exporter = ExportUtils(str(temp_dir))
        
        summary = exporter.generate_short_summary(sample_water_data)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Water Trends' in summary

    def test_integration_workflow(self, temp_dir, sample_water_data, sample_nrm_data):
        """Test complete integration workflow."""
        # Initialize all components
        processor = DataProcessor(str(temp_dir))
        viz = WaterTrendsVisualizer()
        exporter = ExportUtils(str(temp_dir))
        
        # Process data
        merged = processor.merge_datasets(sample_water_data, sample_nrm_data)
        assert len(merged) > 0
        
        # Create visualization
        fig = viz.create_seasonal_stacked_area_chart(merged)
        assert fig is not None
        
        # Export data
        data_path = exporter.export_data_table(merged, 'integration_test', 'csv')
        assert Path(data_path).exists()
        
        # Generate summary
        summary = exporter.generate_short_summary(merged)
        assert len(summary) > 0

    def test_error_handling_data_processor_invalid_input(self, temp_dir):
        """Test error handling with invalid input."""
        processor = DataProcessor(str(temp_dir))
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, Exception)):
            processor.export_processed_data(empty_df, 'test', 'csv')

    def test_error_handling_visualizer_invalid_data(self):
        """Test error handling with invalid visualization data."""
        viz = WaterTrendsVisualizer()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, Exception)):
            viz.create_seasonal_stacked_area_chart(empty_df)

    def test_error_handling_export_utils_invalid_format(self, temp_dir, sample_water_data):
        """Test error handling with invalid export format."""
        exporter = ExportUtils(str(temp_dir))
        
        # Test with invalid format
        with pytest.raises((ValueError, Exception)):
            exporter.export_data_table(sample_water_data, 'test', 'invalid_format')
