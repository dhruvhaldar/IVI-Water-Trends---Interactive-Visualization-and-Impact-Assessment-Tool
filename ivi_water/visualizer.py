"""
Visualization Module

This module provides interactive and static visualizations for water trends
analysis using Plotly and optional 3D visualization with PyVista.
"""

# Standard library imports
import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Optional 3D visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Constants
DEFAULT_CHART_THEME = 'plotly_white'
DEFAULT_CHART_HEIGHT = 600
DEFAULT_CHART_WIDTH = 1000
VALID_THEMES = ['plotly_white', 'plotly_dark', 'plotly', 'ggplot2', 'seaborn', 'simple_white']
SEASON_COLORS = {
    'perennial': '#1f77b4',  # Blue
    'winter': '#ff7f0e',      # Orange
    'monsoon': '#2ca02c',    # Green
    'summer': '#d62728'       # Red
}
SEASON_ORDER = ['perennial', 'winter', 'monsoon', 'summer']

# Logger setup
logger = logging.getLogger(__name__)


class WaterTrendsVisualizer:
    """
    Creates interactive visualizations for water trends analysis.
    
    This class provides comprehensive methods to create various types of charts
    and visualizations for seasonal surface water data and NRM impact assessment.
    It supports 2D visualizations using Plotly and optional 3D visualizations
    using PyVista when available.
    
    Attributes:
        theme (str): Plotly theme for charts
        height (int): Default chart height in pixels
        width (int): Default chart width in pixels
        logger (logging.Logger): Logger instance for this class
        
    Example:
        >>> viz = WaterTrendsVisualizer(theme='plotly_dark', height=800)
        >>> fig = viz.create_seasonal_stacked_area_chart(water_df)
        >>> viz.save_figure(fig, 'water_trends.html')
    """
    
    def __init__(
        self, 
        theme: Optional[str] = None, 
        height: Optional[int] = None, 
        width: Optional[int] = None
    ) -> None:
        """
        Initialize the visualizer with configuration parameters.
        
        This method sets up the visualization environment with specified
        theme, dimensions, and validates the configuration.
        
        Args:
            theme: Plotly theme for charts. Valid options are in VALID_THEMES.
                   If None, uses DEFAULT_CHART_THEME or environment variable.
            height: Default chart height in pixels. If None, uses DEFAULT_CHART_HEIGHT
                   or environment variable.
            width: Default chart width in pixels. If None, uses DEFAULT_CHART_WIDTH
                  or environment variable.
                  
        Raises:
            ValueError: If theme is invalid or dimensions are out of reasonable range
            
        Example:
            >>> viz = WaterTrendsVisualizer(
            ...     theme='plotly_dark', 
            ...     height=800, 
            ...     width=1200
            ... )
        """
        # Validate theme
        theme = theme or os.getenv('DEFAULT_CHART_THEME', DEFAULT_CHART_THEME)
        if theme not in VALID_THEMES:
            raise ValueError(
                f"Invalid theme '{theme}'. Valid options: {VALID_THEMES}"
            )
        
        # Validate dimensions
        height = height or int(os.getenv('CHART_HEIGHT', str(DEFAULT_CHART_HEIGHT)))
        width = width or int(os.getenv('CHART_WIDTH', str(DEFAULT_CHART_WIDTH)))
        
        if not isinstance(height, int) or height < 200 or height > 2000:
            raise ValueError(f"Chart height must be between 200-2000 pixels, got {height}")
        
        if not isinstance(width, int) or width < 400 or width > 4000:
            raise ValueError(f"Chart width must be between 400-4000 pixels, got {width}")
        
        # Set attributes
        self.theme = theme
        self.height = height
        self.width = width
        self.logger = logging.getLogger(__name__)
        
        # Set default theme
        try:
            px.defaults.template = theme
            self.logger.info(f"Set Plotly theme to '{theme}'")
        except Exception as e:
            self.logger.warning(f"Failed to set Plotly theme '{theme}': {e}")
            # Fallback to default theme
            px.defaults.template = DEFAULT_CHART_THEME
            self.theme = DEFAULT_CHART_THEME
        
        self.logger.info(
            f"Initialized WaterTrendsVisualizer: theme='{self.theme}', "
            f"dimensions={self.width}x{self.height}px"
        )
    
    def create_seasonal_stacked_area_chart(
        self, 
        df: pd.DataFrame, 
        location_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive stacked area chart of seasonal water availability.
        
        This method generates a stacked area chart showing water area trends
        across different seasons over time, with support for individual locations
        or aggregated data across all locations.
        
        Args:
            df: DataFrame with water data. Must contain columns: year, season,
                water_area_ha, and optionally location_id.
            location_id: Specific location to plot. If None, aggregates across all
                       locations. Must be a valid location_id from the DataFrame.
            title: Custom chart title. If None, generates default title.
                   
        Returns:
            Plotly Figure object with interactive stacked area chart
            
        Raises:
            ValueError: If DataFrame is empty, missing required columns, or location_id is invalid
            
        Example:
            >>> viz = WaterTrendsVisualizer()
            >>> fig = viz.create_seasonal_stacked_area_chart(
            ...     water_df, location_id='V001',
            ...     title='Water Trends for Village V001'
            ... )
            >>> fig.show()
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_columns = ['year', 'season', 'water_area_ha']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        # Validate location_id if provided
        if location_id is not None:
            if 'location_id' not in df.columns:
                raise ValueError("DataFrame must contain 'location_id' column when location_id is specified")
            
            if location_id not in df['location_id'].values:
                available_locations = df['location_id'].unique().tolist()
                raise ValueError(
                    f"Location '{location_id}' not found in DataFrame. "
                    f"Available locations: {available_locations[:10]}..."
                )
        
        self.logger.info(
            f"Creating seasonal stacked area chart: location_id={location_id}, "
            f"data_points={len(df)}"
        )
        
        try:
            # Filter data based on location_id
            if location_id:
                df_filtered = df[df['location_id'] == location_id].copy()
                title = title or f"Seasonal Water Trends - {location_id}"
                self.logger.debug(f"Filtered data for location {location_id}: {len(df_filtered)} records")
            else:
                # Aggregate across all locations
                df_filtered = df.groupby(['year', 'season'])['water_area_ha'].mean().reset_index()
                title = title or "Average Seasonal Water Trends - All Locations"
                self.logger.debug(f"Aggregated data across {df['location_id'].nunique()} locations")
            
            if df_filtered.empty:
                raise ValueError("No data available after filtering/aggregation")
            
            # Validate data quality
            if (df_filtered['water_area_ha'] < 0).any():
                self.logger.warning("Found negative water area values, these will be displayed as-is")
            
            # Pivot data for stacked area chart
            try:
                pivot_df = df_filtered.pivot(
                    index='year', 
                    columns='season', 
                    values='water_area_ha'
                ).fillna(0)
            except Exception as e:
                self.logger.error(f"Failed to pivot data: {e}")
                raise ValueError(f"Unable to create seasonal chart: {e}")
            
            if pivot_df.empty:
                raise ValueError("No data available after pivoting")
            
            # Create stacked area chart
            fig = go.Figure()
            
            # Add traces for each season in defined order
            available_seasons = [s for s in SEASON_ORDER if s in pivot_df.columns]
            
            if not available_seasons:
                raise ValueError(
                    f"No valid seasons found in data. Available seasons in data: {list(pivot_df.columns)}"
                )
            
            for season in available_seasons:
                season_data = pivot_df[season]
                color = SEASON_COLORS.get(season, '#7f7f7f')  # Default gray if color not defined
                
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=season_data,
                    mode='lines',
                    stackgroup='one',
                    name=season.capitalize(),
                    line=dict(color=color, width=2),
                    fillcolor=color,
                    fill='tonexty',
                    hovertemplate=f'<b>{season.capitalize()}</b><br>' +
                                  'Year: %{x}<br>' +
                                  'Water Area: %{y:.2f} ha<extra></extra>'
                ))
            
            # Update layout with comprehensive styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, family='Arial, sans-serif')
                ),
                xaxis_title=dict(
                    text='Year',
                    font=dict(size=12, family='Arial, sans-serif')
                ),
                yaxis_title=dict(
                    text='Water Area (hectares)',
                    font=dict(size=12, family='Arial, sans-serif')
                ),
                height=self.height,
                width=self.width,
                template=self.theme,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11)
                ),
                hovermode='x unified',
                margin=dict(l=60, r=20, t=80, b=60)
            )
            
            # Add axis formatting
            fig.update_xaxes(
                tickmode='linear',
                tickformat='d',  # Integer format for years
                gridcolor='lightgray',
                gridwidth=1
            )
            
            fig.update_yaxes(
                gridcolor='lightgray',
                gridwidth=1,
                tickformat=',.0f'  # Comma-separated integers
            )
            
            # Log completion
            self.logger.info(
                f"Successfully created seasonal stacked area chart: "
                f"{len(available_seasons)} seasons, {len(pivot_df)} years"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating seasonal stacked area chart: {e}", exc_info=True)
            raise ValueError(f"Failed to create seasonal chart: {e}")
    
    def create_comparison_line_plot(
        self, 
        df: pd.DataFrame, 
        intervention_col: str = 'pond_presence',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create comparative line plots of average water area by intervention presence.
        
        Args:
            df: DataFrame with merged water and NRM data
            intervention_col: Column indicating intervention presence
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if intervention_col not in df.columns:
            raise ValueError(f"Intervention column '{intervention_col}' not found in data")
        
        # Calculate average water area by intervention and year
        avg_data = df.groupby(['year', intervention_col])['water_area_ha'].mean().reset_index()
        avg_data[intervention_col] = avg_data[intervention_col].map({0: 'Without Intervention', 1: 'With Intervention'})
        
        fig = px.line(
            avg_data,
            x='year',
            y='water_area_ha',
            color=intervention_col,
            title=title or "Water Area Trends by Intervention Presence",
            labels={
                'water_area_ha': 'Average Water Area (hectares)',
                'year': 'Year',
                intervention_col: 'Intervention Status'
            },
            markers=True
        )
        
        fig.update_layout(
            height=self.height,
            width=self.width,
            hovermode='x unified'
        )
        
        # Customize colors
        fig.data[0].line.color = '#ff6b6b'  # Red for without intervention
        fig.data[1].line.color = '#51cf66'  # Green for with intervention
        
        return fig
    
    def create_water_body_distribution(
        self, 
        df: pd.DataFrame, 
        season: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create histogram of water body count distribution.
        
        Args:
            df: DataFrame with water data
            season: Specific season to plot. If None, plots all seasons.
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if season:
            df_filtered = df[df['season'] == season]
            title = title or f"Water Body Count Distribution - {season.capitalize()}"
        else:
            df_filtered = df
            title = title or "Water Body Count Distribution - All Seasons"
        
        fig = px.histogram(
            df_filtered,
            x='water_body_count',
            color='season',
            title=title,
            labels={'water_body_count': 'Number of Water Bodies', 'count': 'Frequency'},
            nbins=30,
            barmode='overlay'
        )
        
        fig.update_layout(
            height=self.height,
            width=self.width,
            xaxis_title="Number of Water Bodies",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_trend_heatmap(
        self, 
        df: pd.DataFrame, 
        metric: str = 'water_area_ha',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap of water trends by location and year.
        
        Args:
            df: DataFrame with water data
            metric: Metric to visualize ('water_area_ha' or 'water_body_count')
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Pivot data for heatmap
        heatmap_data = df.pivot_table(
            index='location_id', 
            columns='year', 
            values=metric, 
            aggfunc='mean'
        )
        
        fig = px.imshow(
            heatmap_data,
            title=title or f"Water Trends Heatmap - {metric.replace('_', ' ').title()}",
            labels=dict(x="Year", y="Location", color=metric.replace('_', ' ').title()),
            aspect="auto"
        )
        
        fig.update_layout(
            height=max(self.height, len(heatmap_data) * 20),
            width=self.width
        )
        
        return fig
    
    def create_seasonal_box_plot(
        self, 
        df: pd.DataFrame, 
        year_range: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create box plot of water area by season.
        
        Args:
            df: DataFrame with water data
            year_range: Optional tuple of (start_year, end_year) to filter data
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        df_filtered = df.copy()
        
        if year_range:
            start_year, end_year = year_range
            df_filtered = df_filtered[(df_filtered['year'] >= start_year) & 
                                    (df_filtered['year'] <= end_year)]
        
        fig = px.box(
            df_filtered,
            x='season',
            y='water_area_ha',
            title=title or "Water Area Distribution by Season",
            labels={'water_area_ha': 'Water Area (hectares)', 'season': 'Season'},
            color='season'
        )
        
        fig.update_layout(
            height=self.height,
            width=self.width,
            showlegend=False
        )
        
        return fig
    
    def create_intervention_impact_scatter(
        self, 
        df: pd.DataFrame, 
        intervention_col: str = 'pond_presence',
        impact_col: str = 'crop_yield_ton_per_ha',
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter plot of intervention impact on water area and productivity.
        
        Args:
            df: DataFrame with merged water and NRM data
            intervention_col: Column indicating intervention presence
            impact_col: Column with impact metric (e.g., crop yield)
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if intervention_col not in df.columns:
            raise ValueError(f"Intervention column '{intervention_col}' not found")
        
        if impact_col not in df.columns:
            raise ValueError(f"Impact column '{impact_col}' not found")
        
        # Create intervention labels
        df['intervention_status'] = df[intervention_col].map({0: 'Without Intervention', 1: 'With Intervention'})
        
        fig = px.scatter(
            df,
            x='water_area_ha',
            y=impact_col,
            color='intervention_status',
            title=title or f"Water Area vs {impact_col.replace('_', ' ').title()}",
            labels={
                'water_area_ha': 'Water Area (hectares)',
                impact_col: impact_col.replace('_', ' ').title(),
                'intervention_status': 'Intervention Status'
            },
            hover_data=['location_id', 'year']
        )
        
        fig.update_layout(
            height=self.height,
            width=self.width
        )
        
        return fig
    
    def create_multi_location_dashboard(
        self, 
        df: pd.DataFrame, 
        location_ids: List[str],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard for multiple locations.
        
        Args:
            df: DataFrame with water data
            location_ids: List of location IDs to include
            save_path: Optional path to save HTML dashboard
            
        Returns:
            Plotly Figure object with subplots
        """
        df_filtered = df[df['location_id'].isin(location_ids)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Seasonal Trends', 'Water Body Count', 'Yearly Comparison', 'Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Seasonal trends (stacked area)
        for i, location in enumerate(location_ids[:3]):  # Limit to 3 locations
            loc_data = df_filtered[df_filtered['location_id'] == location]
            pivot_data = loc_data.pivot(index='year', columns='season', values='water_area_ha').fillna(0)
            
            for season in ['perennial', 'winter', 'monsoon']:
                if season in pivot_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pivot_data.index,
                            y=pivot_data[season],
                            mode='lines',
                            stackgroup='one',
                            name=f'{location} - {season}',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # Plot 2: Water body count over time
        for location in location_ids[:3]:
            loc_data = df_filtered[df_filtered['location_id'] == location]
            avg_counts = loc_data.groupby('year')['water_body_count'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=avg_counts.index,
                    y=avg_counts.values,
                    mode='lines+markers',
                    name=f'{location} - Bodies',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Yearly comparison
        yearly_avg = df_filtered.groupby(['year', 'location_id'])['water_area_ha'].mean().reset_index()
        for location in location_ids[:3]:
            loc_data = yearly_avg[yearly_avg['location_id'] == location]
            
            fig.add_trace(
                go.Scatter(
                    x=loc_data['year'],
                    y=loc_data['water_area_ha'],
                    mode='lines+markers',
                    name=location,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Distribution
        fig.add_trace(
            go.Histogram(
                x=df_filtered['water_area_ha'],
                nbinsx=30,
                name='Distribution',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=2 * self.height,
            width=2 * self.width,
            title_text=f"Water Trends Dashboard - {', '.join(location_ids[:3])}",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def create_3d_water_visualization(
        self, 
        df: pd.DataFrame, 
        elevation_data: Optional[Dict] = None,
        location_id: Optional[str] = None
    ) -> Optional['pv.Plotter']:
        """
        Create 3D visualization of water bodies using elevation data.
        
        Args:
            df: DataFrame with water data
            elevation_data: Elevation data from API
            location_id: Specific location to visualize
            
        Returns:
            PyVista Plotter object (if PyVista is available)
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available for 3D visualization")
            return None
        
        if location_id:
            df_filtered = df[df['location_id'] == location_id]
        else:
            df_filtered = df.head(100)  # Limit data for performance
        
        # Create simple 3D visualization
        plotter = pv.Plotter()
        
        # Add water bodies as spheres (simplified representation)
        for _, row in df_filtered.iterrows():
            # Random positions for demonstration
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            z = 0  # Ground level
            
            # Size based on water area
            radius = np.sqrt(row['water_area_ha']) / 10
            
            sphere = pv.Sphere(radius=radius, center=(x, y, z))
            plotter.add_mesh(sphere, color='blue', opacity=0.7)
        
        plotter.add_title(f"3D Water Bodies - {location_id or 'Sample'}")
        plotter.show_grid()
        
        return plotter
    
    def save_figure(self, fig: go.Figure, filename: str, format: str = 'html', **kwargs) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Plotly Figure object
            filename: Output filename
            format: Output format ('html', 'png', 'svg', 'pdf')
            **kwargs: Additional arguments for saving
        """
        output_dir = os.getenv('OUTPUT_DIR', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"{filename}.{format}")
        
        if format == 'html':
            fig.write_html(filepath, **kwargs)
        elif format == 'png':
            fig.write_image(filepath, **kwargs)
        elif format == 'svg':
            fig.write_image(filepath, **kwargs)
        elif format == 'pdf':
            fig.write_image(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Figure saved to {filepath}")


# Utility functions for convenience
def quick_seasonal_plot(df: pd.DataFrame, location_id: Optional[str] = None) -> go.Figure:
    """Quickly create seasonal stacked area chart."""
    viz = WaterTrendsVisualizer()
    return viz.create_seasonal_stacked_area_chart(df, location_id)


def quick_intervention_comparison(df: pd.DataFrame) -> go.Figure:
    """Quickly create intervention comparison plot."""
    viz = WaterTrendsVisualizer()
    return viz.create_comparison_line_plot(df)
