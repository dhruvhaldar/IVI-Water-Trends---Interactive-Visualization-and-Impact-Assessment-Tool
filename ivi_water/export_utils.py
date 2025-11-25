"""
Export Utilities Module

This module provides functions to generate reports, export data,
and create PDF-friendly summaries.
"""

# Standard library imports
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Constants
DEFAULT_OUTPUT_DIR = './outputs'
DEFAULT_EXPORT_DPI = 300
DEFAULT_FIGURE_SIZE = (12, 8)
SUPPORTED_EXPORT_FORMATS = ['csv', 'excel', 'parquet', 'json']
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'pdf', 'svg']
PDF_PAGE_SIZES = {'letter': letter, 'A4': A4}

# Logger setup
logger = logging.getLogger(__name__)


class ExportUtils:
    """
    Utility class for exporting data and generating reports.
    
    This class provides comprehensive methods to export data in various formats,
    generate statistical reports, create visualizations, and produce PDF documents
    with professional formatting and error handling.
    
    Attributes:
        output_dir (Path): Directory for output files
        figure_size (Tuple[int, int]): Default figure dimensions for plots
        dpi (int): Resolution for exported images
        logger (logging.Logger): Logger instance for this class
        
    Example:
        >>> exporter = ExportUtils('./reports')
        >>> filepath = exporter.export_data_table(df, 'water_data', 'excel')
        >>> report_path = exporter.generate_pdf_report(df, 'Water Analysis Report')
    """
    
    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initialize export utilities with configuration.
        
        This method sets up the export environment, creates output directories,
        and configures default settings for data export and report generation.
        
        Args:
            output_dir: Directory for output files. If None, uses OUTPUT_DIR
                       environment variable or DEFAULT_OUTPUT_DIR.
                       
        Raises:
            OSError: If unable to create output directory
            ValueError: If output_dir path is invalid
            
        Example:
            >>> exporter = ExportUtils('./my_reports')
            >>> print(exporter.output_dir)
            PosixPath('./my_reports')
        """
        # Validate and set output directory
        if output_dir is None:
            output_dir = os.getenv('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
        
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("Output directory must be a non-empty string")
        
        try:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {e}")
        
        # Set matplotlib configuration
        try:
            plt.style.use('default')
        except Exception as e:
            logger.warning(f"Failed to set matplotlib style: {e}")
        
        # Set figure and export parameters
        self.figure_size = DEFAULT_FIGURE_SIZE
        self.dpi = int(os.getenv('EXPORT_DPI', str(DEFAULT_EXPORT_DPI)))
        
        # Validate DPI
        if not isinstance(self.dpi, int) or self.dpi < 72 or self.dpi > 600:
            logger.warning(f"DPI {self.dpi} is out of reasonable range, using default")
            self.dpi = DEFAULT_EXPORT_DPI
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"Initialized ExportUtils: output_dir='{self.output_dir}', "
            f"figure_size={self.figure_size}, dpi={self.dpi}"
        )
    
    def export_data_table(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename must be a non-empty string")
        
        # Sanitize filename
        filename = filename.strip()
        filename = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)
        filename = filename.replace(' ', '_')
        
        if not filename:
            raise ValueError("Filename contains no valid characters after sanitization")
        
        # Validate format
        if not isinstance(format, str) or format not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {SUPPORTED_EXPORT_FORMATS}"
            )
        
        self.logger.info(
            f"Exporting DataFrame ({len(df)} rows, {len(df.columns)} columns) "
            f"to {format} format as '{filename}'"
        )
        
        try:
            # Determine file extension and path
            extensions = {
                'csv': '.csv',
                'excel': '.xlsx',
                'parquet': '.parquet',
                'json': '.json'
            }
            
            extension = extensions[format]
            filepath = self.output_dir / f"{filename}{extension}"
            
            # Check if file exists and warn
            if filepath.exists():
                self.logger.warning(f"Overwriting existing file: {filepath}")
            
            # Make a copy to avoid modifying original dataframe
            df_export = df.copy()
            
            # Ensure numeric columns are properly typed and handle None/NA values
            numeric_cols = df_export.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                # Convert column to numeric, coercing errors
                df_export[col] = pd.to_numeric(df_export[col], errors='coerce')
                # If integer dtype and contains NAs, convert to float64 for Excel compatibility
                if pd.api.types.is_integer_dtype(df_export[col]) and df_export[col].isna().any():
                    df_export[col] = df_export[col].astype('float64')
                else:
                    # Otherwise, use nullable Int64 dtype to preserve integer data with NAs
                    df_export[col] = df_export[col].astype('Int64', errors='ignore')
            
            # Export based on format
            if format == 'csv':
                df_export.to_csv(filepath, index=False, encoding='utf-8')
                
            elif format == 'excel':
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Make a deep copy to avoid modifying the original dataframe
                    df_export = df.copy(deep=True)
                    
                    # Ensure all numeric columns are properly typed
                    for col in df_export.select_dtypes(include=['number']).columns:
                        # Convert to float first to handle potential None/NA values
                        df_export[col] = pd.to_numeric(df_export[col], errors='coerce')
                        # Convert to appropriate type
                        if pd.api.types.is_integer_dtype(df[col]):
                            if df_export[col].isna().any():
                                df_export[col] = df_export[col].astype('float64')
                            else:
                                df_export[col] = df_export[col].astype('Int64')
                        else:
                            df_export[col] = df_export[col].astype('float64')
                    
                    # Export to Excel
                    df_export.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Create and export summary sheet
                    try:
                        summary_df = self._create_summary_table(df_export)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    except Exception as e:
                        self.logger.warning(f"Failed to create summary sheet: {e}")
                        
            elif format == 'parquet':
                try:
                    df_export.to_parquet(filepath, index=False, engine='pyarrow')
                except ImportError:
                    # Fallback to fastparquet if pyarrow not available
                    df_export.to_parquet(filepath, index=False, engine='fastparquet')
                    
            elif format == 'json':
                df_export.to_json(filepath, orient='records', indent=2, date_format='iso')
            
            # Verify file was created and get size
            if not filepath.exists():
                raise OSError(f"Failed to create output file: {filepath}")
            
            file_size = filepath.stat().st_size
            
            self.logger.info(
                f"Successfully exported {len(df_export)} rows to {filepath} "
                f"({file_size/1024:.1f} KB)"
            )
            
            return str(filepath)
            
        except PermissionError as e:
            error_msg = f"Permission denied: {filepath if 'filepath' in locals() else 'unknown file'}"
            self.logger.error(error_msg)
            raise PermissionError(error_msg)
        except OSError as e:
            error_msg = f"OS error during export: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg)
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.logger.error(f"Unexpected error during export: {error_msg}", exc_info=True)
            raise ValueError(error_msg) from e
    
    def _create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []
        
        # Basic statistics
        summary_data.append(['Total Records', len(df)])
        
        # Safely get unique locations
        if 'location_id' in df.columns:
            unique_locs = df['location_id'].nunique()
            summary_data.append(['Unique Locations', unique_locs])
        else:
            summary_data.append(['Unique Locations', 'N/A'])
        
        # Safely get year range
        if 'year' in df.columns:
            try:
                years = pd.to_numeric(df['year'].dropna())
                if len(years) > 0:
                    year_range = f"{int(years.min())}-{int(years.max())}"
                    summary_data.append(['Year Range', year_range])
                else:
                    summary_data.append(['Year Range', 'N/A'])
            except Exception:
                summary_data.append(['Year Range', 'N/A'])
        else:
            summary_data.append(['Year Range', 'N/A'])
        
        # Water statistics
        if 'water_area_ha' in df.columns:
            try:
                water_areas = pd.to_numeric(df['water_area_ha'].dropna())
                if len(water_areas) > 0:
                    summary_data.append(['Mean Water Area (ha)', f"{water_areas.mean():.2f}"])
                    summary_data.append(['Max Water Area (ha)', f"{water_areas.max():.2f}"])
                    summary_data.append(['Min Water Area (ha)', f"{water_areas.min():.2f}"])
            except Exception:
                pass  # Skip water stats if there's an error
        
        return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        
    def generate_summary_report(
        self, 
        df: pd.DataFrame, 
        output_dir: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            df: Input DataFrame
            output_dir: Output directory (overrides instance default)
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir
        
        if not filename:
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create PDF report
        if REPORTLAB_AVAILABLE:
            return self._create_pdf_report(df, output_path, filename)
        else:
            # Fallback to HTML report
            return self._create_html_report(df, output_path, filename)
    
    def _create_pdf_report(self, df: pd.DataFrame, output_path: Path, filename: str) -> str:
        """Create PDF report using ReportLab."""
        filepath = output_path / f"{filename}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Water Trends Summary Report", title_style))
        story.append(Spacer(1, 12))
        
        # Date
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Summary statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        summary_df = self._create_summary_table(df)
        
        # Convert DataFrame to table data
        table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        
        # Add charts as images
        if 'water_area_ha' in df.columns and 'year' in df.columns:
            # Create trend chart
            fig, ax = plt.subplots(figsize=self.figure_size)
            yearly_avg = df.groupby('year')['water_area_ha'].mean()
            ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
            ax.set_title('Average Water Area Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Water Area (hectares)')
            ax.grid(True, alpha=0.3)
            
            # Save chart
            chart_path = output_path / f"{filename}_chart.png"
            plt.savefig(chart_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Add to report
            story.append(Paragraph("Water Area Trends", styles['Heading2']))
            story.append(Image(str(chart_path), width=6*inch, height=4*inch))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    def _create_html_report(self, df: pd.DataFrame, output_path: Path, filename: str) -> str:
        """Create HTML report as fallback."""
        filepath = output_path / f"{filename}.html"
        
        # Create summary table
        summary_df = self._create_summary_table(df)
        
        # HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Water Trends Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Water Trends Summary Report</h1>
            <p class="summary">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            {summary_df.to_html(index=False, classes='summary-table')}
            
            <h2>Data Overview</h2>
            <p>Total records: {len(df)}</p>
            <p>Columns: {', '.join(df.columns.tolist())}</p>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return str(filepath)
    
    def generate_short_summary(
        self, 
        df: pd.DataFrame, 
        output_dir: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate Short-friendly summary with key insights.
        
        Args:
            df: Input DataFrame
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to generated summary
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir
        
        if not filename:
            filename = f"short_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = output_path / f"{filename}.txt"
        
        # Extract key insights
        insights = self._extract_key_insights(df)
        
        # Format for Short (emoji-friendly, concise)
        water_area_avg = f"{df['water_area_ha'].mean():.1f} ha" if "water_area_ha" in df.columns else "N/A"
        water_area_max = f"{df['water_area_ha'].max():.1f} ha" if "water_area_ha" in df.columns else "N/A"
        
        ponds_with = df[df['pond_presence'] == 1]['location_id'].nunique() if "pond_presence" in df.columns else "N/A"
        ponds_without = df[df['pond_presence'] == 0]['location_id'].nunique() if "pond_presence" in df.columns else "N/A"
        
        summary_text = f"""📊 *Water Trends Summary Report*
🗓️ Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}

🔍 *Key Insights:*
{insights}

📈 *Quick Stats:*
• Total Locations: {df['location_id'].nunique() if 'location_id' in df.columns else 'N/A'}
• Year Range: {df['year'].min()}-{df['year'].max() if 'year' in df.columns else 'N/A'}
• Total Records: {len(df):,}

💧 *Water Data:*
• Avg Water Area: {water_area_avg}
• Max Water Area: {water_area_max}

🏗️ *Intervention Impact:*
• Locations with Ponds: {ponds_with}
• Locations without Ponds: {ponds_without}

📱 *For detailed analysis, check the full report.*"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        logger.info(f"Short summary generated: {filepath}")
        return str(filepath)
    
    def _extract_key_insights(self, df: pd.DataFrame) -> str:
        """Extract key insights from data."""
        insights = []
        
        # Water area trends
        if 'water_area_ha' in df.columns and 'year' in df.columns:
            yearly_avg = df.groupby('year')['water_area_ha'].mean()
            if len(yearly_avg) > 1:
                trend_slope = np.polyfit(yearly_avg.index, yearly_avg.values, 1)[0]
                if trend_slope > 0.5:
                    insights.append("📈 Water area showing strong increasing trend")
                elif trend_slope < -0.5:
                    insights.append("📉 Water area showing declining trend")
                else:
                    insights.append("➡️ Water area relatively stable")
        
        # Seasonal patterns
        if 'season' in df.columns and 'water_area_ha' in df.columns:
            seasonal_avg = df.groupby('season')['water_area_ha'].mean()
            max_season = seasonal_avg.idxmax()
            min_season = seasonal_avg.idxmin()
            insights.append(f"🌊 {max_season.capitalize()} season has highest water area")
            insights.append(f"🏜️ {min_season.capitalize()} season has lowest water area")
        
        # Intervention impact
        if 'pond_presence' in df.columns and 'water_area_ha' in df.columns:
            with_pond = df[df['pond_presence'] == 1]['water_area_ha'].mean()
            without_pond = df[df['pond_presence'] == 0]['water_area_ha'].mean()
            
            if with_pond > without_pond * 1.2:
                insights.append("💪 Pond locations show 20%+ higher water area")
            elif with_pond < without_pond * 0.8:
                insights.append("⚠️ Pond locations show lower water area")
            else:
                insights.append("⚖️ Similar water levels across pond presence")
        
        # Data quality
        if 'data_quality' in df.columns:
            good_quality = df[df['data_quality'] == 'good'].shape[0]
            quality_pct = (good_quality / len(df)) * 100
            if quality_pct > 90:
                insights.append("✅ High data quality (>90%)")
            elif quality_pct > 70:
                insights.append("⚠️ Moderate data quality (70-90%)")
            else:
                insights.append("❌ Low data quality (<70%)")
        
        return '\n'.join(insights) if insights else "📋 No significant insights detected"
    
    def generate_detailed_report(
        self, 
        df: pd.DataFrame, 
        output_dir: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate detailed report with comprehensive analysis.
        
        Args:
            df: Input DataFrame
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir
        
        if not filename:
            filename = f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create multiple files for detailed report
        report_files = []
        
        # Main summary report
        summary_file = self.generate_summary_report(df, str(output_path), f"{filename}_summary")
        report_files.append(summary_file)
        
        # Data export
        data_file = self.export_data_table(df, f"{filename}_data", 'excel')
        report_files.append(data_file)
        
        # Short summary
        short_file = self.generate_short_summary(df, str(output_path), f"{filename}_short")
        report_files.append(short_file)
        
        # Create index file
        index_file = self._create_report_index(report_files, output_path, filename)
        
        logger.info(f"Detailed report generated with {len(report_files)} files")
        return index_file
    
    def _create_report_index(self, report_files: List[str], output_path: Path, filename: str) -> str:
        """Create an index file for the detailed report."""
        index_path = output_path / f"{filename}_index.txt"
        
        index_content = f"""Water Trends Detailed Report Index
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Report Files:
"""
        
        for i, file_path in enumerate(report_files, 1):
            file_name = Path(file_path).name
            file_type = file_path.split('.')[-1].upper()
            index_content += f"{i}. {file_name} ({file_type})\n"
        
        index_content += f"""
Total Files: {len(report_files)}
Report Directory: {output_path}

For questions or support, contact: IVI Water Trends Team"""
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        return str(index_file)
    
    def create_visualization_exports(
        self, 
        figures: List, 
        filename_prefix: str,
        formats: List[str] = ['png', 'html']
    ) -> List[str]:
        """
        Export multiple visualization figures.
        
        Args:
            figures: List of Plotly Figure objects
            filename_prefix: Prefix for output filenames
            formats: List of export formats
            
        Returns:
            List of exported file paths
        """
        exported_files = []
        
        for i, fig in enumerate(figures):
            base_filename = f"{filename_prefix}_{i+1}"
            
            for format in formats:
                try:
                    if format == 'html':
                        filepath = self.output_dir / f"{base_filename}.html"
                        fig.write_html(str(filepath))
                    elif format in ['png', 'svg', 'pdf']:
                        filepath = self.output_dir / f"{base_filename}.{format}"
                        fig.write_image(str(filepath), width=1200, height=800)
                    
                    exported_files.append(str(filepath))
                    logger.info(f"Figure exported: {filepath}")
                    
                except Exception as e:
                    logger.warning(f"Failed to export figure {i+1} as {format}: {e}")
        
        return exported_files


# Utility functions for convenience
def quick_export(df: pd.DataFrame, filename: str, format: str = 'csv') -> str:
    """Quickly export DataFrame to file."""
    export_utils = ExportUtils()
    return export_utils.export_data_table(df, filename, format)


def quick_summary(df: pd.DataFrame) -> str:
    """Quickly generate summary report."""
    export_utils = ExportUtils()
    return export_utils.generate_summary_report(df)
