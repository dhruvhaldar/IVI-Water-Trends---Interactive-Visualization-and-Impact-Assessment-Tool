"""
Data Processing Module

This module handles data loading, cleaning, merging, and aggregation
for water trends and NRM impact assessment data.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data processing operations for water trends analysis.
    
    This class provides methods to load, clean, merge, and aggregate
    datasets from CoRE Stack APIs and local NRM impact data.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing data files. If None, uses environment or default.
        """
        self.data_dir = Path(data_dir or os.getenv('DATA_DIR', './data'))
        self.processed_data = {}
        
    def load_water_data_from_api(
        self, 
        api_client, 
        location_ids: List[str], 
        start_year: int, 
        end_year: int,
        seasons: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load water data from CoRE Stack API and convert to DataFrame.
        
        Args:
            api_client: CoREStackClient instance
            location_ids: List of location identifiers
            start_year: Start year
            end_year: End year
            seasons: List of seasons to include
            
        Returns:
            DataFrame with water data in long format
        """
        all_data = []
        
        for location_id in location_ids:
            try:
                water_data = api_client.get_seasonal_water_data(
                    location_id, start_year, end_year, seasons
                )
                
                # Convert API response to DataFrame format
                df_data = self._convert_api_response_to_df(water_data, location_id)
                all_data.append(df_data)
                
            except Exception as e:
                logger.warning(f"Failed to load water data for {location_id}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No water data could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return self._clean_water_data(combined_df)
    
    def _convert_api_response_to_df(self, api_data: Dict, location_id: str) -> pd.DataFrame:
        """
        Convert API response data to DataFrame format.
        
        Args:
            api_data: Raw API response data
            location_id: Location identifier
            
        Returns:
            DataFrame in long format
        """
        rows = []
        
        # Handle different API response structures
        if 'timeseries' in api_data:
            timeseries = api_data['timeseries']
        elif 'data' in api_data:
            timeseries = api_data['data']
        else:
            timeseries = api_data
        
        for year_data in timeseries:
            year = year_data.get('year')
            season_data = year_data.get('seasons', {})
            
            for season, water_info in season_data.items():
                row = {
                    'location_id': location_id,
                    'year': year,
                    'season': season,
                    'water_area_ha': water_info.get('area_ha', 0),
                    'water_body_count': water_info.get('count', 0),
                    'data_quality': water_info.get('quality', 'good')
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _clean_water_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate water data.
        
        Args:
            df: Raw water data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert data types
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['water_area_ha'] = pd.to_numeric(df['water_area_ha'], errors='coerce')
        df['water_body_count'] = pd.to_numeric(df['water_body_count'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=['year', 'season', 'water_area_ha'])
        df = df[df['water_area_ha'] >= 0]  # Remove negative areas
        df = df[df['water_body_count'] >= 0]  # Remove negative counts
        
        # Sort data
        df = df.sort_values(['location_id', 'year', 'season'])
        
        logger.info(f"Cleaned water data: {len(df)} records for {df['location_id'].nunique()} locations")
        return df
    
    def load_nrm_impact_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load NRM impact data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, looks for default location.
            
        Returns:
            DataFrame with NRM impact data
        """
        if file_path is None:
            file_path = self.data_dir / 'nrm_impact_data.csv'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NRM impact data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            return self._clean_nrm_data(df)
        except Exception as e:
            logger.error(f"Failed to load NRM impact data: {e}")
            raise
    
    def _clean_nrm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate NRM impact data.
        
        Args:
            df: Raw NRM data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert data types
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        if 'pond_presence' in df.columns:
            df['pond_presence'] = df['pond_presence'].astype(str).str.lower()
            df['pond_presence'] = df['pond_presence'].map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})
            df['pond_presence'] = pd.to_numeric(df['pond_presence'], errors='coerce').fillna(0)
        
        # Remove rows with missing critical data
        required_cols = ['location_id', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in NRM data: {missing_cols}")
        
        df = df.dropna(subset=required_cols)
        
        logger.info(f"Cleaned NRM data: {len(df)} records for {df['location_id'].nunique()} locations")
        return df
    
    def merge_datasets(
        self, 
        water_df: pd.DataFrame, 
        nrm_df: pd.DataFrame,
        merge_on: List[str] = ['location_id', 'year']
    ) -> pd.DataFrame:
        """
        Merge water data with NRM impact data.
        
        Args:
            water_df: Water data DataFrame
            nrm_df: NRM impact data DataFrame
            merge_on: Columns to merge on
            
        Returns:
            Merged DataFrame
        """
        # Ensure merge columns exist
        for col in merge_on:
            if col not in water_df.columns or col not in nrm_df.columns:
                raise ValueError(f"Merge column '{col}' not found in both datasets")
        
        # Merge datasets
        merged_df = pd.merge(water_df, nrm_df, on=merge_on, how='left')
        
        # Add indicators for missing NRM data
        merged_df['nrm_data_available'] = ~merged_df['pond_presence'].isna()
        
        logger.info(f"Merged datasets: {len(merged_df)} records")
        return merged_df
    
    def calculate_water_trends(
        self, 
        df: pd.DataFrame, 
        group_by: List[str] = ['location_id', 'season']
    ) -> pd.DataFrame:
        """
        Calculate water area trends over time.
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by for trend calculation
            
        Returns:
            DataFrame with trend statistics
        """
        trend_stats = []
        
        for group_name, group_df in df.groupby(group_by):
            group_df = group_df.sort_values('year')
            
            # Calculate trend metrics
            water_areas = group_df['water_area_ha'].values
            years = group_df['year'].values
            
            # Linear trend (slope)
            if len(water_areas) > 1 and len(np.unique(years)) > 1:
                try:
                    # Scale years to improve numerical stability
                    years_scaled = (years - np.mean(years)) / np.std(years) if np.std(years) > 0 else years
                    slope = np.polyfit(years_scaled, water_areas, 1, full=False)[0]
                    # Scale the slope back to original units
                    if np.std(years) > 0:
                        slope = slope / np.std(years)
                except (np.RankWarning, np.linalg.LinAlgError):
                    slope = 0.0
            else:
                slope = 0.0
            
            # Summary statistics
            stats = {
                'location_id': group_name[0] if 'location_id' in group_by else None,
                'season': group_name[1] if 'season' in group_by else None,
                'mean_water_area_ha': np.mean(water_areas),
                'std_water_area_ha': np.std(water_areas),
                'min_water_area_ha': np.min(water_areas),
                'max_water_area_ha': np.max(water_areas),
                'trend_slope_ha_per_year': slope,
                'data_points': len(water_areas),
                'start_year': np.min(years),
                'end_year': np.max(years)
            }
            
            trend_stats.append(stats)
        
        return pd.DataFrame(trend_stats)
    
    def aggregate_by_intervention(
        self, 
        df: pd.DataFrame, 
        intervention_col: str = 'pond_presence'
    ) -> pd.DataFrame:
        """
        Aggregate water data by intervention presence.
        
        Args:
            df: Input DataFrame
            intervention_col: Column indicating intervention presence
            
        Returns:
            Aggregated DataFrame
        """
        if intervention_col not in df.columns:
            raise ValueError(f"Intervention column '{intervention_col}' not found")
        
        # Group by intervention presence and calculate statistics
        agg_stats = df.groupby(intervention_col).agg({
            'water_area_ha': ['mean', 'std', 'min', 'max', 'count'],
            'water_body_count': ['mean', 'std'],
            'location_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns]
        agg_stats = agg_stats.reset_index()
        
        # Add intervention labels
        agg_stats['intervention_type'] = agg_stats[intervention_col].map({
            0: 'No Intervention',
            1: 'With Intervention'
        })
        
        return agg_stats
    
    def create_seasonal_summary(
        self, 
        df: pd.DataFrame, 
        location_level: str = 'location_id'
    ) -> pd.DataFrame:
        """
        Create seasonal summary statistics.
        
        Args:
            df: Input DataFrame
            location_level: Level for spatial aggregation
            
        Returns:
            Seasonal summary DataFrame
        """
        # Group by location and season
        seasonal_summary = df.groupby([location_level, 'season']).agg({
            'water_area_ha': ['mean', 'std', 'min', 'max'],
            'water_body_count': ['mean', 'std'],
            'year': ['min', 'max', 'count']
        }).round(2)
        
        # Flatten column names
        seasonal_summary.columns = ['_'.join(col).strip() for col in seasonal_summary.columns]
        seasonal_summary = seasonal_summary.reset_index()
        
        return seasonal_summary
    
    def export_processed_data(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        format: str = 'csv'
    ) -> None:
        """
        Export processed data to file.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            format: Export format ('csv', 'excel', 'parquet')
        """
        output_path = self.data_dir / 'processed' / filename
        
        # Create processed directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        elif format == 'excel':
            df.to_excel(output_path.with_suffix('.xlsx'), index=False)
        elif format == 'parquet':
            df.to_parquet(output_path.with_suffix('.parquet'), index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported data to {output_path}")


# Utility functions for convenience
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sample datasets for testing and demonstration.
    
    Returns:
        Tuple of (water_data, nrm_data) DataFrames
    """
    processor = DataProcessor()
    
    # Create sample water data
    water_data = pd.DataFrame({
        'location_id': ['V001', 'V001', 'V001', 'V002', 'V002', 'V002'] * 3,
        'year': [2020, 2021, 2022] * 6,
        'season': ['perennial', 'winter', 'monsoon'] * 6,
        'water_area_ha': np.random.uniform(10, 100, 18),
        'water_body_count': np.random.randint(1, 20, 18),
        'data_quality': ['good'] * 18
    })
    
    # Create sample NRM data
    nrm_data = pd.DataFrame({
        'location_id': ['V001', 'V002'],
        'year': [2021, 2021],
        'pond_presence': [1, 0],
        'crop_yield_ton_per_ha': [2.5, 1.8],
        'drought_sensitivity': ['low', 'high']
    })
    
    return water_data, nrm_data
