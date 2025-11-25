import pytest
import pandas as pd
import plotly.graph_objects as go

from ivi_water.visualizer import WaterTrendsVisualizer


@pytest.fixture
def sample_df():
    data = {
        'year': [2018, 2018, 2019, 2019],
        'season': ['perennial', 'winter', 'perennial', 'winter'],
        'water_area_ha': [10, 5, 12, 6],
        'location_id': ['loc1', 'loc1', 'loc1', 'loc1']
    }
    df = pd.DataFrame(data)
    df['year'] = df['year'].astype(int)
    return df


@pytest.fixture
def sample_df_multiple_locations():
    data = {
        'year': [2018, 2018, 2019, 2019, 2018, 2018, 2019, 2019],
        'season': ['perennial', 'winter', 'perennial', 'winter', 'perennial', 'winter', 'perennial', 'winter'],
        'water_area_ha': [10, 5, 12, 6, 9, 4, 11, 5],
        'location_id': ['loc1', 'loc1', 'loc1', 'loc1', 'loc2', 'loc2', 'loc2', 'loc2'],
        'pond_presence': [1, 1, 1, 1, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    df['year'] = df['year'].astype(int)
    return df


def test_init_with_valid_params():
    viz = WaterTrendsVisualizer(theme='plotly_dark', height=500, width=900)
    assert viz.theme == 'plotly_dark'
    assert viz.height == 500
    assert viz.width == 900


def test_init_invalid_theme():
    with pytest.raises(ValueError):
        WaterTrendsVisualizer(theme='invalid_theme')


def test_init_invalid_height():
    with pytest.raises(ValueError):
        WaterTrendsVisualizer(height=100)  # too small


def test_init_invalid_width():
    with pytest.raises(ValueError):
        WaterTrendsVisualizer(width=300)  # too small


def test_create_seasonal_stacked_area_chart_empty_df():
    viz = WaterTrendsVisualizer(height=600, width=900)
    with pytest.raises(ValueError):
        viz.create_seasonal_stacked_area_chart(pd.DataFrame())


# def test_create_seasonal_stacked_area_chart_missing_columns(sample_df):
#     viz = WaterTrendsVisualizer(height=600, width=900)
#     df = sample_df.drop(columns=['season'])
#     with pytest.raises(ValueError):
#         viz.create_seasonal_stacked_area_chart(df)


def test_create_seasonal_stacked_area_chart_invalid_location_id(sample_df):
    viz = WaterTrendsVisualizer(height=600, width=900)
    with pytest.raises(ValueError):
        viz.create_seasonal_stacked_area_chart(sample_df, location_id='invalid_loc')


# def test_create_seasonal_stacked_area_chart_aggregated(sample_df):
#     viz = WaterTrendsVisualizer(theme='plotly_white', height=600, width=900)
#     fig = viz.create_seasonal_stacked_area_chart(sample_df)
#     assert isinstance(fig, go.Figure)


# def test_create_seasonal_stacked_area_chart_filtered(sample_df):
#     viz = WaterTrendsVisualizer(height=600, width=900)
#     fig = viz.create_seasonal_stacked_area_chart(sample_df, location_id='loc1')
#     assert isinstance(fig, go.Figure)
#     assert all(trace.name.lower() in ['perennial', 'winter'] for trace in fig.data)


def test_create_comparison_line_plot_missing_intervention_col(sample_df):
    viz = WaterTrendsVisualizer(height=600, width=900)
    with pytest.raises(ValueError):
        viz.create_comparison_line_plot(sample_df, intervention_col='nonexistent')


# def test_create_comparison_line_plot(sample_df_multiple_locations):
#     viz = WaterTrendsVisualizer(height=600, width=900)
#     fig = viz.create_comparison_line_plot(sample_df_multiple_locations, intervention_col='pond_presence')
#     assert isinstance(fig, go.Figure)
#     colors = [trace.line.color for trace in fig.data]
#     assert '#ff6b6b' in colors  # color for 'Without Intervention'
#     assert '#51cf66' in colors  # color for 'With Intervention'


# def test_create_seasonal_stacked_area_chart_negative_values():
#     viz = WaterTrendsVisualizer(height=600, width=900)
#     data = {
#         'year': [2018, 2019],
#         'season': ['perennial', 'perennial'],
#         'water_area_ha': [-10, 20],
#         'location_id': ['loc1', 'loc1']
#     }
#     df = pd.DataFrame(data)
#     df['year'] = df['year'].astype(int)
#     fig = viz.create_seasonal_stacked_area_chart(df, location_id='loc1')
#     assert isinstance(fig, go.Figure)


# def test_save_figure(tmp_path):
#     viz = WaterTrendsVisualizer(height=600, width=900)
#     fig = viz.create_seasonal_stacked_area_chart(pd.DataFrame({
#         'year': [2018, 2019],
#         'season': ['perennial', 'perennial'],
#         'water_area_ha': [10, 20],
#         'location_id': ['loc1', 'loc1']
#     }), location_id='loc1')

#     filename = tmp_path / "test_chart.html"
#     viz.save_figure(fig, str(filename), format='html')
#     assert filename.exists()

#     with pytest.raises(ValueError):
#         viz.save_figure(fig, str(tmp_path / "test_chart.badext"), format='badext')
