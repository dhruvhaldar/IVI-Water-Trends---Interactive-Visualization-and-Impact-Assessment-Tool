#!/usr/bin/env python3
"""
IVI Water Trends - Getting Started Demo

This script demonstrates the core functionality of the IVI Water Trends tool
for analyzing seasonal surface water trends and NRM impact assessment.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import IVI Water modules
from ivi_water import CoREStackClient, DataProcessor, WaterTrendsVisualizer, ExportUtils

def main():
    """Main demonstration function."""
    print("🌊 IVI Water Trends - Getting Started Demo")
    print("=" * 50)
    
    # Verify installation
    print("\n✅ Checking imports...")
    print(f"📁 Data directory: {os.getenv('DATA_DIR', './data')}")
    print(f"📤 Output directory: {os.getenv('OUTPUT_DIR', './outputs')}")
    
    # Load sample data
    print("\n📊 Loading sample data...")
    from ivi_water.data_processor import load_sample_data
    
    water_df, nrm_df = load_sample_data()
    
    print(f"Water data shape: {water_df.shape}")
    print(f"Locations: {water_df['location_id'].unique()}")
    print(f"Seasons: {water_df['season'].unique()}")
    print(f"Years: {water_df['year'].unique()}")
    
    print(f"\nNRM data shape: {nrm_df.shape}")
    
    # Data processing
    print("\n🔗 Processing and merging data...")
    processor = DataProcessor()
    merged_df = processor.merge_datasets(water_df, nrm_df)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Records with NRM data: {merged_df['nrm_data_available'].sum()}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    viz = WaterTrendsVisualizer()
    
    # Seasonal trends
    print("Creating seasonal trends chart...")
    seasonal_fig = viz.create_seasonal_stacked_area_chart(water_df)
    viz.save_figure(seasonal_fig, "demo_seasonal_trends", "html")
    
    # Intervention comparison
    print("Creating intervention comparison...")
    comparison_fig = viz.create_comparison_line_plot(merged_df)
    viz.save_figure(comparison_fig, "demo_intervention_comparison", "html")
    
    # Water body distribution
    print("Creating water body distribution...")
    distribution_fig = viz.create_water_body_distribution(water_df)
    viz.save_figure(distribution_fig, "demo_distribution", "html")
    
    # Statistical analysis
    print("\n📊 Performing statistical analysis...")
    trends_df = processor.calculate_water_trends(water_df)
    print("Water trends analysis:")
    print(trends_df.head())
    
    intervention_agg = processor.aggregate_by_intervention(merged_df)
    print("\nIntervention impact summary:")
    print(intervention_agg)
    
    # Generate reports
    print("\n📋 Generating reports...")
    export_utils = ExportUtils()
    
    # Summary report
    summary_report = export_utils.generate_summary_report(merged_df, filename="demo_summary")
    print(f"Summary report: {summary_report}")
    
    # Short summary
    short_summary = export_utils.generate_short_summary(merged_df, filename="demo_short")
    print(f"Short summary: {short_summary}")
    
    # Display Short summary content
    print("\n📱 Short Summary Content:")
    print("-" * 30)
    with open(short_summary, 'r', encoding='utf-8') as f:
        print(f.read())
    
    # API demo (if API key is available)
    print("\n🔗 Testing API connection...")
    try:
        client = CoREStackClient()
        print("✅ API client initialized successfully")
        
        # Get spatial units (this will work with real API)
        units = client.get_spatial_units(unit_type="village")
        print(f"📍 Found {len(units)} villages")
        
        if units:
            units_df = pd.DataFrame(units[:5])
            print("Sample spatial units:")
            print(units_df)
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("Please check your API key in the .env file")
    
    # CLI examples
    print("\n🖥️ CLI Usage Examples:")
    print("1. Get spatial units:")
    print("   ivi-water get-spatial-units --unit-type village --state Maharashtra")
    print("\n2. Fetch water data:")
    print("   ivi-water fetch-water-data --locations V001,V002 --start-year 2020 --end-year 2022")
    print("\n3. Merge data:")
    print("   ivi-water merge-data --water-data water.csv --nrm-data nrm.csv")
    print("\n4. Create visualizations:")
    print("   ivi-water visualize --data merged.csv --chart-type seasonal")
    print("\n5. Generate reports:")
    print("   ivi-water generate-report --data merged.csv --report-type summary")
    
    print("\n🎉 Demo completed successfully!")
    print("Check the outputs directory for generated files.")

if __name__ == "__main__":
    main()
