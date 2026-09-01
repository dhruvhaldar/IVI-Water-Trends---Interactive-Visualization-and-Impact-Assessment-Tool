"""
Command Line Interface Module

This module provides a CLI for batch processing and report generation
using the IVI Water Trends tool.
"""

# Standard library imports
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Third-party imports
import click
import pandas as pd
from dotenv import load_dotenv

# Local imports
from .api_client import CoREStackClient
from .data_processor import DataProcessor
from .visualizer import WaterTrendsVisualizer
from .export_utils import ExportUtils, sanitize_dataframe, sanitize_filename
from .security_utils import sanitize_for_terminal

# Constants
DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_UNIT_TYPE = "village"
DEFAULT_SPATIAL_UNITS_OUTPUT = "spatial_units.csv"
DEFAULT_WATER_DATA_OUTPUT = "water_data.csv"
DEFAULT_NRM_DATA_OUTPUT = "nrm_data.csv"
DEFAULT_TRENDS_OUTPUT = "water_trends.csv"
DEFAULT_REPORT_OUTPUT = "water_trends_report.html"

VALID_UNIT_TYPES = [
    "village",
    "micro-watershed",
    "tehsil",
    "watershed",
    "block",
    "district",
]
VALID_SEASONS = ["perennial", "winter", "monsoon", "summer"]

# Load environment variables
load_dotenv()


# Setup logging with better configuration
def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        verbose: Enable debug logging if True

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ivi_water_trends.log"),
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logging()


@click.group()
@click.version_option(version="0.1.0", prog_name="IVI Water Trends")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging with debug information",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Data directory path. Must exist and be a directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory path. Will be created if it does not exist.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: bool,
    data_dir: Optional[str],
    output_dir: Optional[str],
) -> None:
    """
    IVI Water Trends - Interactive Visualization and Impact Assessment Tool.
    
    This CLI tool provides comprehensive water trends analysis for villages and
    watersheds, including data fetching, processing, visualization, and reporting.
    
    Global Options:
        verbose: Enable detailed debug logging
        data-dir: Directory containing input data files
        output-dir: Directory for generated outputs
        
    Example:
        $ ivi-water --verbose --output-dir ./results fetch-water-data \
            --locations V001,V002 --start-year 2020 --end-year 2022
    """
    try:
        # Ensure context object exists
        ctx.ensure_object(dict)

        # Setup logging based on verbose flag
        global logger
        if verbose:
            logger = setup_logging(verbose=True)
            logger.debug("Verbose logging enabled")

        # Store configuration in context
        ctx.obj["verbose"] = verbose

        # Validate and set data directory
        if data_dir:
            data_path = Path(data_dir).resolve()
            if not data_path.exists():
                raise click.ClickException(
                    f"Data directory does not exist: {data_path}"
                )
            if not data_path.is_dir():
                raise click.ClickException(f"Data path is not a directory: {data_path}")
            ctx.obj["data_dir"] = str(data_path)
        else:
            ctx.obj["data_dir"] = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

        # Validate and set output directory
        if output_dir:
            output_path = Path(output_dir).resolve()
        else:
            output_path = Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

        # Create output directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            ctx.obj["output_dir"] = str(output_path)
        except OSError as e:
            raise click.ClickException(
                f"Failed to create output directory {output_path}: {e}"
            )

        # Log configuration
        logger.info(f"Data directory: {ctx.obj['data_dir']}")
        logger.info(f"Output directory: {ctx.obj['output_dir']}")
        logger.info(f"Verbose mode: {'enabled' if verbose else 'disabled'}")

    except Exception as e:
        logger.error(f"CLI initialization failed: {e}")
        raise click.ClickException(f"CLI initialization failed: {e}")


@cli.command()
@click.option(
    "--unit-type",
    default=DEFAULT_UNIT_TYPE,
    type=click.Choice(VALID_UNIT_TYPES, case_sensitive=False),
    help=f"Spatial unit type. Options: {VALID_UNIT_TYPES}",
)
@click.option("--state", help="State filter (optional). Specify state name or code.")
@click.option(
    "--output",
    default=DEFAULT_SPATIAL_UNITS_OUTPUT,
    help="Output filename for spatial units CSV",
)
@click.pass_context
def get_spatial_units(
    ctx: click.Context, unit_type: str, state: Optional[str], output: str
) -> None:
    """
    Fetch available spatial units from CoRE Stack.

    This command retrieves spatial units (villages, watersheds, etc.) from the
    CoRE Stack API and saves them to a CSV file for further analysis.

    Examples:
        $ ivi-water get-spatial-units --unit-type village --state Maharashtra
        $ ivi-water get-spatial-units --unit-type micro-watershed --output my_units.csv
    """
    try:
        logger.info(
            f"Fetching spatial units with unit_type='{unit_type}', state='{state}'"
        )

        # Validate output filename
        if not output or not isinstance(output, str):
            raise click.ClickException("Output filename must be a non-empty string")

        try:
            output = sanitize_filename(output)
        except ValueError as e:
            raise click.ClickException(f"Invalid output filename: {e}")

        if not output.endswith(".csv"):
            output += ".csv"

        # Initialize API client
        try:
            client = CoREStackClient()
        except ValueError as e:
            raise click.ClickException(f"API client initialization failed: {e}")

        # Fetch spatial units
        click.echo(
            click.style(
                f"⏳ Fetching {unit_type} spatial units...",
                fg="blue",
            )
        )
        try:
            units = client.get_spatial_units(unit_type, state)
        except Exception as e:
            logger.error(f"Failed to fetch spatial units: {e}")
            raise click.ClickException(f"Failed to fetch spatial units: {e}")

        if not units:
            click.echo(
                click.style(
                    "⚠️ No spatial units found. Try a different unit type or state.",
                    fg="yellow",
                )
            )
            logger.warning(
                f"No spatial units found for unit_type='{unit_type}', state='{state}'"
            )
            return

        # Convert to DataFrame and validate
        try:
            df = pd.DataFrame(units)

            if df.empty:
                click.echo(
                    click.style(
                        "⚠️ No valid spatial unit data found. Try a different unit type or state.",
                        fg="yellow",
                    )
                )
                return

            # Validate required columns
            required_cols = ["id", "name"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"Missing expected columns in spatial units: {missing_cols}"
                )

        except Exception as e:
            logger.error(f"Failed to create DataFrame from spatial units: {e}")
            raise click.ClickException(f"Failed to process spatial units data: {e}")

        # Save to file
        try:
            output_path = Path(ctx.obj["output_dir"]) / output
            # Sanitize data before saving
            sanitize_dataframe(df).to_csv(output_path, index=False, encoding="utf-8")

            file_size = output_path.stat().st_size
            logger.info(
                f"Saved {len(df):,} spatial units to {output_path} ({file_size:,} bytes)"
            )

        except Exception as e:
            logger.error(f"Failed to save spatial units to file: {e}")
            raise click.ClickException(f"Failed to save output file: {e}")

        # Display results
        click.echo(click.style(f"✅ Found {len(df):,} {unit_type}s", fg="green"))
        click.echo(click.style(f"✅ Saved to {output_path}", fg="green"))

        # Show sample data
        if len(df) > 0:
            click.echo("\n" + click.style("Sample data:", fg="blue", bold=True))

            # Add explicit tabular headers
            click.echo(click.style(f"  {'ID':<15} | {'Name':<30} | State", fg="cyan"))
            click.echo(
                click.style(f"  {'-'*15}-+-{'-'*30}-+-{'-'*15}", fg="cyan", dim=True)
            )

            # Display first few rows with formatting
            sample_size = min(5, len(df))
            sample_df = df.head(sample_size)

            # Format for better display
            for idx, row in sample_df.iterrows():
                safe_id = sanitize_for_terminal(str(row.get("id", "N/A")))
                safe_name = sanitize_for_terminal(str(row.get("name", "N/A")))
                safe_state = sanitize_for_terminal(str(row.get("state", "N/A")))
                click.echo(
                    f"  {safe_id:<15} | {safe_name:<30} | "
                    + click.style(safe_state, dim=True)
                )

            if len(df) > sample_size:
                click.echo(
                    click.style(
                        f"  ... and {len(df) - sample_size:,} more entries", dim=True
                    )
                )

        click.echo(
            "\n"
            + click.style(f"💡 Tip: Use these location IDs with ", fg="yellow")
            + click.style("ivi-water fetch-water-data", fg="cyan", bold=True)
        )

        # Log completion
        logger.info(
            f"Successfully completed spatial units fetch: {len(df):,} units retrieved"
        )

    except click.ClickException:
        # Re-raise Click exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_spatial_units: {e}", exc_info=True)
        raise click.ClickException(f"Unexpected error: {e}")


@cli.command()
@click.option(
    "--locations",
    required=True,
    help="Comma-separated location IDs (e.g., V001,V002,V003)",
)
@click.option(
    "--start-year",
    required=True,
    type=int,
    help="Start year for data collection (inclusive)",
)
@click.option(
    "--end-year",
    required=True,
    type=int,
    help="End year for data collection (inclusive)",
)
@click.option("--seasons", help=f"Comma-separated seasons. Options: {VALID_SEASONS}")
@click.option(
    "--output",
    default=DEFAULT_WATER_DATA_OUTPUT,
    help="Output filename for water data CSV",
)
@click.pass_context
def fetch_water_data(
    ctx: click.Context,
    locations: str,
    start_year: int,
    end_year: int,
    seasons: Optional[str],
    output: str,
) -> None:
    """
    Fetch seasonal water data from CoRE Stack.

    This command retrieves seasonal surface water data for specified locations
    and time periods from the CoRE Stack API and saves it to a CSV file.

    Examples:
        $ ivi-water fetch-water-data --locations V001,V002 --start-year 2020 --end-year 2022
        $ ivi-water fetch-water-data --locations V001 --start-year 2020 --end-year 2022 --seasons monsoon,winter
    """
    try:
        logger.info(
            f"Starting water data fetch: locations={locations}, years={start_year}-{end_year}"
        )

        # Validate and parse locations
        if not locations or not isinstance(locations, str):
            raise click.ClickException(
                "Locations parameter is required and must be a string"
            )

        location_list = [loc.strip() for loc in locations.split(",") if loc.strip()]

        if not location_list:
            raise click.ClickException("No valid location IDs provided")

        # Validate years
        current_year = datetime.now().year
        if start_year < 1900 or start_year > current_year + 5:
            raise click.ClickException(
                f"Start year {start_year} is out of reasonable range"
            )

        if end_year < 1900 or end_year > current_year + 5:
            raise click.ClickException(
                f"End year {end_year} is out of reasonable range"
            )

        if start_year > end_year:
            raise click.ClickException(
                "Start year must be less than or equal to end year"
            )

        # Validate and parse seasons
        season_list = None
        if seasons:
            if not isinstance(seasons, str):
                raise click.ClickException("Seasons parameter must be a string")

            season_list = [s.strip().lower() for s in seasons.split(",") if s.strip()]

            invalid_seasons = [s for s in season_list if s not in VALID_SEASONS]
            if invalid_seasons:
                raise click.ClickException(
                    f"Invalid seasons: {invalid_seasons}. "
                    f"Valid options: {VALID_SEASONS}"
                )

        # Validate output filename
        if not output or not isinstance(output, str):
            raise click.ClickException("Output filename must be a non-empty string")

        try:
            output = sanitize_filename(output)
        except ValueError as e:
            raise click.ClickException(f"Invalid output filename: {e}")

        if not output.endswith(".csv"):
            output += ".csv"

        # Initialize clients
        try:
            client = CoREStackClient()
            processor = DataProcessor(ctx.obj["data_dir"])
        except ValueError as e:
            raise click.ClickException(f"Client initialization failed: {e}")

        # Display progress
        click.echo(
            click.style(
                f"⏳ Fetching water data for {len(location_list):,} locations...",
                fg="blue",
            )
        )

        # Fetch water data
        try:
            water_df = processor.load_water_data_from_api(
                client, location_list, start_year, end_year, season_list
            )

        except Exception as e:
            logger.error(f"Failed to fetch water data: {e}")
            raise click.ClickException(f"Failed to fetch water data: {e}")

        # Validate results
        if water_df.empty:
            click.echo(
                click.style(
                    "⚠️ No water data found for the specified parameters. Try adjusting the year range or seasons.",
                    fg="yellow",
                )
            )
            logger.warning("No water data retrieved")
            return

        # Save to file
        try:
            output_path = Path(ctx.obj["output_dir"]) / output
            # Sanitize data before saving
            sanitize_dataframe(water_df).to_csv(
                output_path, index=False, encoding="utf-8"
            )

            file_size = output_path.stat().st_size
            logger.info(
                f"Saved {len(water_df):,} water records to {output_path} ({file_size:,} bytes)"
            )

        except Exception as e:
            logger.error(f"Failed to save water data: {e}")
            raise click.ClickException(f"Failed to save output file: {e}")

        # Display results and summary
        click.echo(click.style(f"✅ Fetched {len(water_df):,} records", fg="green"))
        click.echo(click.style(f"✅ Saved to {output_path}", fg="green"))

        click.echo("\n" + click.style("Data Summary:", fg="blue", bold=True))
        click.echo(
            click.style(f"  Locations: {water_df['location_id'].nunique():,}", dim=True)
        )
        click.echo(
            click.style(
                f"  Years: {water_df['year'].min()} - {water_df['year'].max()}",
                dim=True,
            )
        )
        click.echo(
            click.style(
                f"  Seasons: {', '.join(sorted(water_df['season'].unique()))}", dim=True
            )
        )
        click.echo(
            click.style(
                f"  Total water area range: {water_df['water_area_ha'].min():,.2f} - {water_df['water_area_ha'].max():,.2f} ha",
                dim=True,
            )
        )

        # Show sample data
        if len(water_df) > 0:
            click.echo("\n" + click.style("Sample records:", fg="blue"))

            # Add explicit tabular headers
            click.echo(
                click.style(
                    f"  {'Location':<8} | {'Year':<6} | {'Season':<10} | {'Water Area':<11} | Bodies",
                    fg="cyan",
                )
            )
            click.echo(
                click.style(
                    f"  {'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}",
                    fg="cyan",
                    dim=True,
                )
            )

            sample_size = min(3, len(water_df))
            sample_df = water_df.head(sample_size)

            for idx, row in sample_df.iterrows():
                safe_loc = sanitize_for_terminal(str(row["location_id"]))
                safe_season = sanitize_for_terminal(str(row["season"]))
                click.echo(
                    f"  {safe_loc:<8} | {row['year']:<6} | {safe_season:<10} | "
                    f"{row['water_area_ha']:<8,.2f} ha | "
                    + click.style(f"{row['water_body_count']:<3,} bodies", dim=True)
                )

            if len(water_df) > sample_size:
                click.echo(
                    click.style(
                        f"  ... and {len(water_df) - sample_size:,} more records",
                        dim=True,
                    )
                )

        click.echo(
            "\n"
            + click.style(f"💡 Tip: Visualize these trends with ", fg="yellow")
            + click.style(
                f"ivi-water visualize --data {output_path}", fg="cyan", bold=True
            )
        )

        # Log completion
        logger.info(
            f"Successfully completed water data fetch: {len(water_df):,} records "
            f"for {len(location_list):,} locations from {start_year}-{end_year}"
        )

    except click.ClickException:
        # Re-raise Click exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fetch_water_data: {e}", exc_info=True)
        raise click.ClickException(f"Unexpected error: {e}")


@cli.command()
@click.option(
    "--water-data",
    required=True,
    type=click.Path(exists=True),
    help="Water data CSV file",
)
@click.option(
    "--nrm-data", type=click.Path(exists=True), help="NRM impact data CSV file"
)
@click.option("--output", default="merged_data.csv", help="Output filename")
@click.pass_context
def merge_data(ctx, water_data, nrm_data, output):
    """Merge water data with NRM impact data"""
    try:
        click.echo(
            click.style(
                "⏳ Merging water and NRM data...",
                fg="blue",
            )
        )

        processor = DataProcessor(ctx.obj["data_dir"])

        # Load water data
        water_df = processor.load_csv_safe(water_data)

        if water_df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before merging.",
                    fg="yellow",
                )
            )
            return

        water_df = processor._clean_water_data(water_df)

        # Load NRM data if provided
        if nrm_data:
            nrm_df = processor.load_nrm_impact_data(nrm_data)
            merged_df = processor.merge_datasets(water_df, nrm_df)
        else:
            merged_df = water_df
            merged_df["nrm_data_available"] = False

        # Save merged data
        try:
            output = sanitize_filename(output)
        except ValueError as e:
            raise click.ClickException(f"Invalid output filename: {e}")

        output_path = Path(ctx.obj["output_dir"]) / output
        # Sanitize data before saving
        sanitize_dataframe(merged_df).to_csv(output_path, index=False)

        click.echo(click.style(f"✅ Merged data saved to {output_path}", fg="green"))
        click.echo(click.style(f"  Total records: {len(merged_df):,}", dim=True))

        if nrm_data:
            click.echo(
                click.style(
                    f"  Records with NRM data: {merged_df['nrm_data_available'].sum():,}",
                    dim=True,
                )
            )

        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Create a comprehensive dashboard with ", fg="yellow"
            )
            + click.style(
                f"ivi-water dashboard --data {output_path}", fg="cyan", bold=True
            )
        )

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input data CSV file"
)
@click.option("--location-id", help="Specific location to visualize")
@click.option(
    "--chart-type",
    default="seasonal",
    type=click.Choice(["seasonal", "comparison", "distribution", "heatmap"]),
    help="Type of chart to create",
)
@click.option("--output", help="Output filename (without extension)")
@click.option(
    "--format",
    default="html",
    type=click.Choice(["html", "png", "svg"]),
    help="Output format",
)
@click.pass_context
def visualize(ctx, data, location_id, chart_type, output, format):
    """Create visualizations from water data"""
    try:
        processor = DataProcessor(ctx.obj["data_dir"])
        df = processor.load_csv_safe(data)

        if df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before visualizing.",
                    fg="yellow",
                )
            )
            return

        viz = WaterTrendsVisualizer()

        click.echo(
            click.style(
                f"⏳ Generating {chart_type} chart...",
                fg="blue",
            )
        )

        # Generate filename if not provided
        if not output:
            location_suffix = f"_{location_id}" if location_id else ""
            output = f"{chart_type}_chart{location_suffix}"

        try:
            output = sanitize_filename(output)
        except ValueError as e:
            raise click.ClickException(f"Invalid output filename: {e}")

        # Create appropriate chart
        if chart_type == "seasonal":
            fig = viz.create_seasonal_stacked_area_chart(df, location_id)
        elif chart_type == "comparison":
            fig = viz.create_comparison_line_plot(df)
        elif chart_type == "distribution":
            fig = viz.create_water_body_distribution(df)
        elif chart_type == "heatmap":
            fig = viz.create_trend_heatmap(df)

        # Save figure
        viz.save_figure(fig, output, format, output_dir=ctx.obj["output_dir"])

        click.echo(
            click.style(
                f"✅ Chart saved to {ctx.obj['output_dir']}/{output}.{format}",
                fg="green",
            )
        )

        click.echo(
            "\n"
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        )

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input data CSV file"
)
@click.option("--locations", help="Comma-separated location IDs for dashboard")
@click.option("--output", default="dashboard", help="Output filename")
@click.pass_context
def dashboard(ctx, data, locations, output):
    """Create comprehensive dashboard for multiple locations"""
    try:
        processor = DataProcessor(ctx.obj["data_dir"])
        df = processor.load_csv_safe(data)

        if df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before creating dashboard.",
                    fg="yellow",
                )
            )
            return

        viz = WaterTrendsVisualizer()

        location_list = (
            [loc.strip() for loc in locations.split(",")] if locations else None
        )

        if location_list:
            df_filtered = df[df["location_id"].isin(location_list)]
        else:
            # Use top 5 locations by data volume
            location_counts = df["location_id"].value_counts().head(5)
            location_list = location_counts.index.tolist()
            df_filtered = df[df["location_id"].isin(location_list)]

        if df_filtered.empty:
            click.echo(
                click.style(
                    "⚠️ No data found for the specified locations. Please check the location IDs.",
                    fg="yellow",
                )
            )
            return

        click.echo(
            click.style(
                "⏳ Generating comprehensive dashboard...",
                fg="blue",
            )
        )

        fig = viz.create_multi_location_dashboard(df_filtered, location_list)

        # Save dashboard using secure save_figure method
        viz.save_figure(fig, output, "html", output_dir=ctx.obj["output_dir"])

        click.echo(
            click.style(
                f"✅ Dashboard saved to {ctx.obj['output_dir']}/{output}.html",
                fg="green",
            )
        )
        click.echo(
            click.style(f"  Included locations: {', '.join(location_list)}", dim=True)
        )

        click.echo(
            "\n"
            + click.style(f"💡 Tip: Generate a full report with ", fg="yellow")
            + click.style(
                f"ivi-water generate-report --data {data}", fg="cyan", bold=True
            )
        )

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Input data CSV file"
)
@click.option(
    "--report-type",
    default="summary",
    type=click.Choice(["summary", "detailed", "short"]),
    help="Type of report to generate",
)
@click.option("--output", help="Output filename")
@click.pass_context
def generate_report(ctx, data, report_type, output):
    """Generate reports from water data"""
    try:
        processor = DataProcessor(ctx.obj["data_dir"])
        df = processor.load_csv_safe(data)

        if df.empty:
            click.echo(
                click.style(
                    "⚠️ No data found in the input file. Please ensure the CSV is populated before generating reports.",
                    fg="yellow",
                )
            )
            return

        export_utils = ExportUtils()

        if not output:
            output = f"{report_type}_report"

        try:
            output = sanitize_filename(output)
        except ValueError as e:
            raise click.ClickException(f"Invalid output filename: {e}")

        click.echo(
            click.style(
                f"⏳ Generating {report_type} report...",
                fg="blue",
            )
        )

        if report_type == "summary":
            output_path = export_utils.generate_summary_report(
                df, ctx.obj["output_dir"], output
            )
        elif report_type == "detailed":
            output_path = export_utils.generate_detailed_report(
                df, ctx.obj["output_dir"], output
            )
        elif report_type == "short":
            output_path = export_utils.generate_short_summary(
                df, ctx.obj["output_dir"], output
            )

        click.echo(click.style(f"✅ Report saved to {output_path}", fg="green"))

        click.echo(
            "\n"
            + click.style(
                f"💡 Tip: Open the generated report file in your browser to view the insights.",
                fg="yellow",
            )
        )

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--template", type=click.Choice(["basic", "advanced"]), help="Notebook template"
)
@click.option("--output-dir", type=click.Path(), help="Output directory for notebooks")
@click.option(
    "--force", is_flag=True, help="Overwrite existing notebooks without prompting"
)
@click.pass_context
def setup_notebooks(ctx, template, output_dir, force):
    """Setup Jupyter notebooks for analysis"""
    try:
        click.echo(
            click.style(
                "⏳ Setting up notebooks...",
                fg="blue",
            )
        )

        notebook_dir = Path(output_dir) if output_dir else Path("notebooks")
        notebook_dir.mkdir(exist_ok=True)

        created = False
        skipped = False

        if template == "basic":
            res = create_basic_notebook(notebook_dir, force)
            if res:
                created = True
            else:
                skipped = True
        elif template == "advanced":
            res = create_advanced_notebook(notebook_dir, force)
            if res:
                created = True
            else:
                skipped = True
        else:
            res1 = create_basic_notebook(notebook_dir, force)
            res2 = create_advanced_notebook(notebook_dir, force)
            if res1 or res2:
                created = True
            if not res1 or not res2:
                skipped = True

        if created:
            click.echo(
                click.style(
                    f"✅ Notebooks created/updated in {notebook_dir}", fg="green"
                )
            )
        if skipped:
            click.echo(
                click.style(
                    f"⚠️ Some notebooks were skipped to prevent overwriting.",
                    fg="yellow",
                )
            )

        if created:
            click.echo(
                "\n"
                + click.style(
                    f"💡 Tip: Start Jupyter notebook to begin your analysis. ",
                    fg="yellow",
                )
                + click.style("jupyter notebook", fg="cyan", bold=True)
            )

    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg="red"), err=True)
        sys.exit(1)


def create_basic_notebook(notebook_dir: Path, force: bool = False) -> bool:
    """Create basic analysis notebook. Returns True if created/overwritten, False if skipped."""
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
                    "## Setup",
                ],
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
                    "viz = WaterTrendsVisualizer()",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## Load Data"]},
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
                    'print(f"Water data: {water_df.shape}")\n',
                    'print(f"NRM data: {nrm_df.shape}")',
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Data Exploration"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Display basic statistics\n",
                    'print("Water Data Summary:")\n',
                    "print(water_df.describe())\n",
                    "\n",
                    'print("\\nNRM Data Summary:")\n',
                    "print(nrm_df.describe())",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## Visualizations"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create seasonal trends chart\n",
                    "fig = viz.create_seasonal_stacked_area_chart(water_df)\n",
                    "fig.show()",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    import json

    notebook_path = notebook_dir / "basic_analysis.ipynb"

    if notebook_path.exists() and not force:
        if not click.confirm(
            click.style(
                f"⚠️  Notebook {notebook_path.name} already exists. Overwrite and lose any changes?",
                fg="yellow",
            ),
            default=False,
        ):
            return False

    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    return True


def create_advanced_notebook(notebook_dir: Path, force: bool = False) -> bool:
    """Create advanced analysis notebook. Returns True if created/overwritten, False if skipped."""
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
                    "## Setup",
                ],
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
                    "export_utils = ExportUtils()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Data Loading and Processing"],
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
                    'print(f"Merged data: {merged_df.shape}")\n',
                    'print(f"Trends data: {trends_df.shape}")\n',
                    'print(f"Aggregated data: {agg_df.shape}")',
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Advanced Visualizations"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create comprehensive dashboard\n",
                    "locations = merged_df['location_id'].unique()[:3]\n",
                    "dashboard_fig = viz.create_multi_location_dashboard(merged_df, locations.tolist())\n",
                    "dashboard_fig.show()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Statistical Analysis"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Calculate correlation between interventions and water availability\n",
                    "correlation = merged_df[['pond_presence', 'water_area_ha']].corr()\n",
                    'print("Correlation matrix:")\n',
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
                    "print(\"Average trend without intervention:\", no_intervention_trends['trend_slope_ha_per_year'].mean())",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    import json

    notebook_path = notebook_dir / "advanced_analysis.ipynb"

    if notebook_path.exists() and not force:
        if not click.confirm(
            click.style(
                f"⚠️  Notebook {notebook_path.name} already exists. Overwrite and lose any changes?",
                fg="yellow",
            ),
            default=False,
        ):
            return False

    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    return True


def main():
    """Main entry point for CLI"""
    cli()


if __name__ == "__main__":
    main()
