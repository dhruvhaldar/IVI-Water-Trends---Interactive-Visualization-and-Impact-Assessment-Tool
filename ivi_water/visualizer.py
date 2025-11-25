"""
Visualization Module

This module provides interactive and static visualizations for water trends
analysis using Plotly and optional 3D visualization with PyVista.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
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

logger = logging.getLogger(__name__)


class WaterTrendsVisualizer:
    """
    Creates interactive visualizations for water trends analysis.
    
    This class provides methods to create various types of charts and
    visualizations for seasonal surface water data and NRM impact assessment.
    """
    
    def __init__(self, theme: Optional[str] = None, height: Optional[int] = None, width: Optional[int] = None):
        """
        Initialize the visualizer.
        
        Args:
            theme: Plotly theme ('plotly_white', 'plotly_dark', etc.)
            height: Default chart height
            width: Default chart width
        """
        self.theme = theme or os.getenv('DEFAULT_CHART_THEME', 'plotly_white')
        self.height = height or int(os.getenv('CHART_HEIGHT', '600'))
        self.width = width or int(os.getenv('CHART_WIDTH', '1000'))
        
        # Set default theme
        px.defaults.template = self.theme
    
    def create_seasonal_stacked_area_chart(
        self, 
        df: pd.DataFrame, 
        location_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive stacked area chart of seasonal water availability.
        
        Args:
            df: DataFrame with water data
            location_id: Specific location to plot. If None, plots all locations.
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if location_id:
            df_filtered = df[df['location_id'] == location_id]
            title = title or f"Seasonal Water Trends - {location_id}"
        else:
            df_filtered = df.groupby(['year', 'season'])['water_area_ha'].mean().reset_index()
            title = title or "Average Seasonal Water Trends - All Locations"
        
        # Pivot data for stacked area chart
        pivot_df = df_filtered.pivot(index='year', columns='season', values='water_area_ha').fillna(0)
        
        # Create stacked area chart
        fig = go.Figure()
        
        seasons = ['perennial', 'winter', 'monsoon']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        
        for i, season in enumerate(seasons):
            if season in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[season],
                    mode='lines',
                    stackgroup='one',
                    name=season.capitalize(),
                    line=dict(color=colors[i]),
                    fillcolor=colors[i]
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Water Area (hectares)",
            height=self.height,
            width=self.width,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
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
