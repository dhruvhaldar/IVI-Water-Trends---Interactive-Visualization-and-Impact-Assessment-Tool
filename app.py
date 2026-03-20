import streamlit as st
import os
import io
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ivi_water import CoREStackClient, DataProcessor, WaterTrendsVisualizer, ExportUtils

# Set up page config
st.set_page_config(
    page_title="IVI Water Trends",
    page_icon="🌊",
    layout="wide"
)

st.title("IVI Water Trends Dashboard")
st.markdown("Analyze seasonal surface water trends using CoRE Stack data.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Check if API Key exists
    api_key_exists = bool(os.getenv("CORE_API_KEY"))
    if not api_key_exists:
        st.warning("CORE_API_KEY not found in environment. You can enter it below for this session.")
        api_key_input = st.text_input("CoRE Stack API Key", type="password")
        if api_key_input:
            os.environ["CORE_API_KEY"] = api_key_input
            api_key_exists = True
    else:
        st.success("API Key configured.")
        
    st.divider()
    
    locations_input = st.text_input("Locations (comma-separated)", value="V001, V002", help="e.g., V001, V002")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2000, max_value=2050, value=2020)
    with col2:
        end_year = st.number_input("End Year", min_value=2000, max_value=2050, value=2022)
        
    theme = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "seaborn", "ggplot2", "none"])
    
    analyze_btn = st.button("Fetch & Analyze Data", type="primary", use_container_width=True)

# --- Main App Logic ---
if analyze_btn:
    if not api_key_exists and 'api_key_input' not in locals():
        st.error("Please provide a valid CoRE Stack API Key in the sidebar or via the .env file.")
        st.stop()
        
    # Process inputs
    locations = [loc.strip() for loc in locations_input.split(",") if loc.strip()]
    if not locations:
        st.error("Please provide at least one valid location.")
        st.stop()
        
    if start_year > end_year:
        st.error("Start Year cannot be greater than End Year.")
        st.stop()

    with st.spinner("Initializing components..."):
        try:
            client = CoREStackClient()
            processor = DataProcessor('./data')
            viz = WaterTrendsVisualizer(theme=theme, height=600)
            exporter = ExportUtils('./outputs/reports') # Ensure directory exists or let ExportUtils handle it
            
            # Create directories if they don't exist
            os.makedirs('./data', exist_ok=True)
            os.makedirs('./outputs/reports', exist_ok=True)
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            st.stop()

    with st.spinner(f"Fetching data for {len(locations)} locations..."):
        try:
            water_data = processor.load_water_data_from_api(
                client, locations, start_year, end_year
            )
            st.success("Data fetched successfully!")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

    st.subheader("Seasonal Stacked Area Chart")
    with st.spinner("Generating visualization..."):
        try:
             fig = viz.create_seasonal_stacked_area_chart(water_data)
             st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
             st.error(f"Error generating chart: {str(e)}")

    st.divider()
    st.subheader("Data Export")
    
    # Show dataframe preview
    with st.expander("View Raw Data Preview"):
        st.dataframe(water_data.head(100))
        
    # Download buttons
    col3, col4 = st.columns(2)
    
    # CSV Download
    csv = water_data.to_csv(index=False).encode('utf-8')
    with col3:
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"water_trends_data_{start_year}_{end_year}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # Optionally, we could provide Excel export via a physical file or BytesIO
    # Try generating an Excel file in memory
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            water_data.to_excel(writer, index=False, sheet_name='Water Trends')
        
        with col4:
            st.download_button(
                label="Download Data as Excel",
                data=buffer.getvalue(),
                file_name=f"water_trends_data_{start_year}_{end_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except:
        pass
