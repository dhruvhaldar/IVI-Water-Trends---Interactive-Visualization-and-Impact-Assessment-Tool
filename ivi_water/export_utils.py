"""
Export Utilities Module

This module provides functions to generate reports, export data,
and create Short-friendly summaries.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

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

logger = logging.getLogger(__name__)


class ExportUtils:
    """
    Utility class for exporting data and generating reports.
    
    This class provides methods to export data in various formats
    and generate comprehensive reports including Short-friendly summaries.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize export utilities.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir or os.getenv('OUTPUT_DIR', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        self.figure_size = (12, 8)
        self.dpi = int(os.getenv('EXPORT_DPI', '300'))
    
    def export_data_table(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        format: str = 'csv'
    ) -> str:
        """
        Export data table to various formats.
        
        Args:
            df: DataFrame to export
            filename: Output filename (without extension)
            format: Export format ('csv', 'excel', 'parquet')
            
        Returns:
            Path to exported file
        """
        if format == 'csv':
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            filepath = self.output_dir / f"{filename}.xlsx"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add summary sheet
                summary_df = self._create_summary_table(df)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        elif format == 'parquet':
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to {filepath}")
        return str(filepath)
    
    def _create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []
        
        # Basic statistics
        summary_data.append(['Total Records', len(df)])
        summary_data.append(['Unique Locations', df['location_id'].nunique() if 'location_id' in df.columns else 'N/A'])
        summary_data.append(['Year Range', f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A'])
        
        # Water statistics
        if 'water_area_ha' in df.columns:
            summary_data.append(['Mean Water Area (ha)', f"{df['water_area_ha'].mean():.2f}"])
            summary_data.append(['Max Water Area (ha)', f"{df['water_area_ha'].max():.2f}"])
            summary_data.append(['Min Water Area (ha)', f"{df['water_area_ha'].min():.2f}"])
        
        # Intervention statistics
        if 'pond_presence' in df.columns:
            with_pond = df[df['pond_presence'] == 1].shape[0]
            summary_data.append(['Records with Pond', with_pond])
            summary_data.append(['Records without Pond', len(df) - with_pond])
        
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
