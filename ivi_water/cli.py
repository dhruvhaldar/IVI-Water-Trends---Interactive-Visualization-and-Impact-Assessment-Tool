"""
Command Line Interface Module

This module provides a CLI for batch processing and report generation
using the IVI Water Trends tool.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
from dotenv import load_dotenv

from .api_client import CoREStackClient
from .data_processor import DataProcessor
from .visualizer import WaterTrendsVisualizer
from .export_utils import ExportUtils

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--data-dir', type=click.Path(exists=True), help='Data directory path')
@click.option('--output-dir', type=click.Path(), help='Output directory path')
@click.pass_context
def cli(ctx, verbose, data_dir, output_dir):
    """IVI Water Trends - Interactive Visualization and Impact Assessment Tool"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if data_dir:
        ctx.obj['data_dir'] = data_dir
    else:
        ctx.obj['data_dir'] = os.getenv('DATA_DIR', './data')
    
    if output_dir:
        ctx.obj['output_dir'] = output_dir
    else:
        ctx.obj['output_dir'] = os.getenv('OUTPUT_DIR', './outputs')
    
    # Create output directory
    os.makedirs(ctx.obj['output_dir'], exist_ok=True)


@cli.command()
@click.option('--unit-type', default='village', help='Spatial unit type')
@click.option('--state', help='State filter')
@click.option('--output', default='spatial_units.csv', help='Output filename')
@click.pass_context
def get_spatial_units(ctx, unit_type, state, output):
    """Fetch available spatial units from CoRE Stack"""
    try:
        client = CoREStackClient()
        units = client.get_spatial_units(unit_type, state)
        
        if not units:
            click.echo("No spatial units found.")
            return
        
        df = pd.DataFrame(units)
        output_path = Path(ctx.obj['output_dir']) / output
        df.to_csv(output_path, index=False)
        
        click.echo(f"Found {len(units)} {unit_type}s")
        click.echo(f"Saved to {output_path}")
        
        # Show sample
        click.echo("\nSample data:")
        click.echo(df.head().to_string())
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--locations', required=True, help='Comma-separated location IDs')
@click.option('--start-year', required=True, type=int, help='Start year')
@click.option('--end-year', required=True, type=int, help='End year')
@click.option('--seasons', help='Comma-separated seasons (perennial,winter,monsoon)')
@click.option('--output', default='water_data.csv', help='Output filename')
@click.pass_context
def fetch_water_data(ctx, locations, start_year, end_year, seasons, output):
    """Fetch seasonal water data from CoRE Stack"""
    try:
        location_list = [loc.strip() for loc in locations.split(',')]
        season_list = [s.strip() for s in seasons.split(',')] if seasons else None
        
        client = CoREStackClient()
        processor = DataProcessor(ctx.obj['data_dir'])
        
        click.echo(f"Fetching water data for {len(location_list)} locations...")
        
        water_df = processor.load_water_data_from_api(
            client, location_list, start_year, end_year, season_list
        )
        
        output_path = Path(ctx.obj['output_dir']) / output
        water_df.to_csv(output_path, index=False)
        
        click.echo(f"Fetched {len(water_df)} records")
        click.echo(f"Saved to {output_path}")
        
        # Show summary
        click.echo(f"\nSummary:")
        click.echo(f"Locations: {water_df['location_id'].nunique()}")
        click.echo(f"Years: {water_df['year'].min()} - {water_df['year'].max()}")
        click.echo(f"Seasons: {', '.join(water_df['season'].unique())}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--water-data', required=True, type=click.Path(exists=True), help='Water data CSV file')
@click.option('--nrm-data', type=click.Path(exists=True), help='NRM impact data CSV file')
@click.option('--output', default='merged_data.csv', help='Output filename')
@click.pass_context
def merge_data(ctx, water_data, nrm_data, output):
    """Merge water data with NRM impact data"""
    try:
        processor = DataProcessor(ctx.obj['data_dir'])
        
        # Load water data
        water_df = pd.read_csv(water_data)
        water_df = processor._clean_water_data(water_df)
        
        # Load NRM data if provided
        if nrm_data:
            nrm_df = processor.load_nrm_impact_data(nrm_data)
            merged_df = processor.merge_datasets(water_df, nrm_df)
        else:
            merged_df = water_df
            merged_df['nrm_data_available'] = False
        
        # Save merged data
        output_path = Path(ctx.obj['output_dir']) / output
        merged_df.to_csv(output_path, index=False)
        
        click.echo(f"Merged data saved to {output_path}")
        click.echo(f"Total records: {len(merged_df)}")
        
        if nrm_data:
            click.echo(f"Records with NRM data: {merged_df['nrm_data_available'].sum()}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', required=True, type=click.Path(exists=True), help='Input data CSV file')
@click.option('--location-id', help='Specific location to visualize')
@click.option('--chart-type', default='seasonal', 
              type=click.Choice(['seasonal', 'comparison', 'distribution', 'heatmap']),
              help='Type of chart to create')
@click.option('--output', help='Output filename (without extension)')
@click.option('--format', default='html', type=click.Choice(['html', 'png', 'svg']),
              help='Output format')
@click.pass_context
def visualize(ctx, data, location_id, chart_type, output, format):
    """Create visualizations from water data"""
    try:
        df = pd.read_csv(data)
        viz = WaterTrendsVisualizer()
        
        # Generate filename if not provided
        if not output:
            location_suffix = f"_{location_id}" if location_id else ""
            output = f"{chart_type}_chart{location_suffix}"
        
        # Create appropriate chart
        if chart_type == 'seasonal':
            fig = viz.create_seasonal_stacked_area_chart(df, location_id)
        elif chart_type == 'comparison':
            fig = viz.create_comparison_line_plot(df)
        elif chart_type == 'distribution':
            fig = viz.create_water_body_distribution(df)
        elif chart_type == 'heatmap':
            fig = viz.create_trend_heatmap(df)
        
        # Save figure
        viz.save_figure(fig, output, format)
        
        click.echo(f"Chart saved to {ctx.obj['output_dir']}/{output}.{format}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', required=True, type=click.Path(exists=True), help='Input data CSV file')
@click.option('--locations', help='Comma-separated location IDs for dashboard')
@click.option('--output', default='dashboard', help='Output filename')
@click.pass_context
def dashboard(ctx, data, locations, output):
    """Create comprehensive dashboard for multiple locations"""
    try:
        df = pd.read_csv(data)
        viz = WaterTrendsVisualizer()
        
        location_list = [loc.strip() for loc in locations.split(',')] if locations else None
        
        if location_list:
            df_filtered = df[df['location_id'].isin(location_list)]
        else:
            # Use top 5 locations by data volume
            location_counts = df['location_id'].value_counts().head(5)
            location_list = location_counts.index.tolist()
            df_filtered = df[df['location_id'].isin(location_list)]
        
        fig = viz.create_multi_location_dashboard(df_filtered, location_list)
        
        output_path = Path(ctx.obj['output_dir']) / f"{output}.html"
        fig.write_html(output_path)
        
        click.echo(f"Dashboard saved to {output_path}")
        click.echo(f"Included locations: {', '.join(location_list)}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', required=True, type=click.Path(exists=True), help='Input data CSV file')
@click.option('--report-type', default='summary', 
              type=click.Choice(['summary', 'detailed', 'short']),
              help='Type of report to generate')
@click.option('--output', help='Output filename')
@click.pass_context
def generate_report(ctx, data, report_type, output):
    """Generate reports from water data"""
    try:
        df = pd.read_csv(data)
        export_utils = ExportUtils()
        
        if not output:
            output = f"{report_type}_report"
        
        if report_type == 'summary':
            output_path = export_utils.generate_summary_report(
                df, ctx.obj['output_dir'], output
            )
        elif report_type == 'detailed':
            output_path = export_utils.generate_detailed_report(
                df, ctx.obj['output_dir'], output
            )
        elif report_type == 'short':
            output_path = export_utils.generate_short_summary(
                df, ctx.obj['output_dir'], output
            )
        
        click.echo(f"Report saved to {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--template', type=click.Choice(['basic', 'advanced']), help='Notebook template')
@click.option('--output-dir', type=click.Path(), help='Output directory for notebooks')
@click.pass_context
def setup_notebooks(ctx, template, output_dir):
    """Setup Jupyter notebooks for analysis"""
    try:
        notebook_dir = Path(output_dir) if output_dir else Path('notebooks')
        notebook_dir.mkdir(exist_ok=True)
        
        if template == 'basic':
            create_basic_notebook(notebook_dir)
        elif template == 'advanced':
            create_advanced_notebook(notebook_dir)
        else:
            create_basic_notebook(notebook_dir)
            create_advanced_notebook(notebook_dir)
        
        click.echo(f"Notebooks created in {notebook_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def create_basic_notebook(notebook_dir: Path):
    """Create basic analysis notebook"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# IVI Water Trends - Basic Analysis\n",
                    "\n",
                    "This notebook provides a basic analysis of seasonal surface water trends.\n",
                    "\n",
                    "## Setup"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import plotly.express as px\n",
                    "from ivi_water import CoREStackClient, DataProcessor, WaterTrendsVisualizer\n",
                    "\n",
                    "# Initialize components\n",
                    "client = CoREStackClient()\n",
                    "processor = DataProcessor()\n",
                    "viz = WaterTrendsVisualizer()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Load Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load your data here\n",
                    "# water_df = pd.read_csv('path/to/water_data.csv')\n",
                    "# nrm_df = pd.read_csv('path/to/nrm_data.csv')\n",
                    "\n",
                    "# For demonstration, create sample data\n",
                    "from ivi_water.data_processor import load_sample_data\n",
                    "water_df, nrm_df = load_sample_data()\n",
                    "\n",
                    "print(f\"Water data: {water_df.shape}\")\n",
                    "print(f\"NRM data: {nrm_df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Data Exploration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Display basic statistics\n",
                    "print(\"Water Data Summary:\")\n",
                    "print(water_df.describe())\n",
                    "\n",
                    "print(\"\\nNRM Data Summary:\")\n",
                    "print(nrm_df.describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create seasonal trends chart\n",
                    "fig = viz.create_seasonal_stacked_area_chart(water_df)\n",
                    "fig.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    notebook_path = notebook_dir / "basic_analysis.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)


def create_advanced_notebook(notebook_dir: Path):
    """Create advanced analysis notebook"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# IVI Water Trends - Advanced Analysis\n",
                    "\n",
                    "This notebook provides advanced analysis including trend calculations,\n",
                    "impact assessment, and comprehensive visualizations.\n",
                    "\n",
                    "## Setup"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import plotly.graph_objects as go\n",
                    "from plotly.subplots import make_subplots\n",
                    "from ivi_water import CoREStackClient, DataProcessor, WaterTrendsVisualizer, ExportUtils\n",
                    "\n",
                    "# Initialize components\n",
                    "client = CoREStackClient()\n",
                    "processor = DataProcessor()\n",
                    "viz = WaterTrendsVisualizer()\n",
                    "export_utils = ExportUtils()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Data Loading and Processing"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load and merge data\n",
                    "from ivi_water.data_processor import load_sample_data\n",
                    "water_df, nrm_df = load_sample_data()\n",
                    "merged_df = processor.merge_datasets(water_df, nrm_df)\n",
                    "\n",
                    "# Calculate trends\n",
                    "trends_df = processor.calculate_water_trends(merged_df)\n",
                    "agg_df = processor.aggregate_by_intervention(merged_df)\n",
                    "\n",
                    "print(f\"Merged data: {merged_df.shape}\")\n",
                    "print(f\"Trends data: {trends_df.shape}\")\n",
                    "print(f\"Aggregated data: {agg_df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Advanced Visualizations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create comprehensive dashboard\n",
                    "locations = merged_df['location_id'].unique()[:3]\n",
                    "dashboard_fig = viz.create_multi_location_dashboard(merged_df, locations.tolist())\n",
                    "dashboard_fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Statistical Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Calculate correlation between interventions and water availability\n",
                    "correlation = merged_df[['pond_presence', 'water_area_ha']].corr()\n",
                    "print(\"Correlation matrix:\")\n",
                    "print(correlation)\n",
                    "\n",
                    "# Trend analysis by intervention status\n",
                    "intervention_trends = processor.calculate_water_trends(\n",
                    "    merged_df[merged_df['pond_presence'] == 1]\n",
                    ")\n",
                    "no_intervention_trends = processor.calculate_water_trends(\n",
                    "    merged_df[merged_df['pond_presence'] == 0]\n",
                    ")\n",
                    "\n",
                    "print(\"\\nAverage trend with intervention:\", intervention_trends['trend_slope_ha_per_year'].mean())\n",
                    "print(\"Average trend without intervention:\", no_intervention_trends['trend_slope_ha_per_year'].mean())"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    notebook_path = notebook_dir / "advanced_analysis.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)


def main():
    """Main entry point for CLI"""
    cli()


if __name__ == '__main__':
    main()
