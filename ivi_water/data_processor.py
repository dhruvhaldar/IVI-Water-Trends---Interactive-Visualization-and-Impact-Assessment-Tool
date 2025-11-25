"""
Data Processing Module

This module handles data loading, cleaning, merging, and aggregation
for water trends and NRM impact assessment data.
"""

# Standard library imports
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Third-party imports
import pandas as pd
import numpy as np

# Constants
DEFAULT_DATA_DIR = './data'
DEFAULT_OUTPUT_DIR = './outputs'
MIN_DATA_POINTS_FOR_TREND = 2
MAX_WATER_AREA_HA = 10000.0  # Maximum reasonable water area in hectares
MIN_WATER_AREA_HA = 0.0
VALID_SEASONS = ['perennial', 'winter', 'monsoon', 'summer']
VALID_INTERVENTION_TYPES = ['pond', 'check_dam', 'contour_bund', 'other']

# Logger setup
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data processing operations for water trends analysis.
    
    This class provides methods to load, clean, merge, and aggregate
    datasets from CoRE Stack APIs and local NRM impact data.
    
    Attributes:
        data_dir (Path): Directory containing data files
        processed_data (Dict[str, pd.DataFrame]): Cache for processed datasets
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, data_dir: Optional[str] = None) -> None:
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing data files. If None, uses environment or default.
            
        Raises:
            ValueError: If data directory doesn't exist and cannot be created
        """
        self.data_dir = Path(data_dir or os.getenv('DATA_DIR', DEFAULT_DATA_DIR))
        
        # Ensure data directory exists
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create data directory {self.data_dir}: {e}")
            
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)
        
    def load_water_data_from_api(
        self, 
        api_client: Any,  # CoREStackClient instance
        location_ids: List[str], 
        start_year: int, 
        end_year: int,
        seasons: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load water data from CoRE Stack API and convert to DataFrame.
        
        This method fetches seasonal surface water data for multiple locations
        from the CoRE Stack API, validates the response, and converts it to a
        standardized DataFrame format.
        
        Args:
            api_client: CoREStackClient instance for API communication
            location_ids: List of location identifiers to fetch data for
            start_year: Start year for data collection (inclusive)
            end_year: End year for data collection (inclusive)
            seasons: List of seasons to include. If None, uses default seasons.
            
        Returns:
            DataFrame with water data in long format containing columns:
            location_id, year, season, water_area_ha
            
        Raises:
            ValueError: If no data can be loaded or input parameters are invalid
            TypeError: If api_client doesn't have required methods
            
        Example:
            >>> client = CoREStackClient()
            >>> processor = DataProcessor()
            >>> df = processor.load_water_data_from_api(
            ...     client, ['V001', 'V002'], 2020, 2022, ['monsoon', 'winter']
            ... )
            >>> print(df.columns.tolist())
            ['location_id', 'year', 'season', 'water_area_ha']
        """
        # Input validation
        if not location_ids:
            raise ValueError("location_ids cannot be empty")
            
        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")
            
        if seasons is None:
            seasons = VALID_SEASONS
        else:
            invalid_seasons = [s for s in seasons if s not in VALID_SEASONS]
            if invalid_seasons:
                raise ValueError(f"Invalid seasons: {invalid_seasons}")
        
        # Validate api_client has required method
        if not hasattr(api_client, 'get_seasonal_water_data'):
            raise TypeError("api_client must have get_seasonal_water_data method")
        
        self.logger.info(
            f"Loading water data for {len(location_ids)} locations "
            f"from {start_year} to {end_year} for seasons: {seasons}"
        )
        
        all_data: List[pd.DataFrame] = []
        successful_locations = 0
        
        for location_id in location_ids:
            try:
                self.logger.debug(f"Fetching data for location: {location_id}")
                
                water_data = api_client.get_seasonal_water_data(
                    location_id, start_year, end_year, seasons
                )
                
                # Validate API response
                if not water_data:
                    self.logger.warning(f"No data returned for location {location_id}")
                    continue
                
                # Convert API response to DataFrame format
                df_data = self._convert_api_response_to_df(water_data, location_id)
                
                if not df_data.empty:
                    all_data.append(df_data)
                    successful_locations += 1
                    self.logger.debug(f"Successfully loaded {len(df_data)} records for {location_id}")
                else:
                    self.logger.warning(f"Empty DataFrame created for location {location_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load water data for {location_id}: {e}", exc_info=True)
                continue
        
        if not all_data:
            raise ValueError(
                f"No water data could be loaded from {len(location_ids)} locations. "
                "Check API connection and location IDs."
            )
        
        self.logger.info(
            f"Successfully loaded data for {successful_locations}/{len(location_ids)} locations"
        )
        
        try:
            combined_df = pd.concat(all_data, ignore_index=True)
            return self._clean_water_data(combined_df)
        except Exception as e:
            self.logger.error(f"Failed to combine DataFrames: {e}", exc_info=True)
            raise ValueError(f"Error combining water data: {e}")
    
    def _convert_api_response_to_df(self, api_data: Dict[str, Any], location_id: str) -> pd.DataFrame:
        """
        Convert API response data to DataFrame format.
        
        This method handles different API response structures and converts
        them to a standardized DataFrame format with proper validation.
        
        Args:
            api_data: Raw API response data containing timeseries information
            location_id: Location identifier for the data
            
        Returns:
            DataFrame in long format with columns: location_id, year, season,
            water_area_ha, water_body_count, data_quality
            
        Raises:
            ValueError: If API data structure is invalid or missing required fields
            
        Example:
            >>> api_data = {
            ...     'timeseries': [
            ...         {
            ...             'year': 2020,
            ...             'seasons': {
            ...                 'monsoon': {'area_ha': 100.5, 'count': 5}
            ...             }
            ...         }
            ...     ]
            ... }
            >>> df = processor._convert_api_response_to_df(api_data, 'V001')
            >>> print(df[['location_id', 'year', 'season', 'water_area_ha']].values.tolist())
            [['V001', 2020, 'monsoon', 100.5]]
        """
        if not api_data:
            raise ValueError("API data cannot be empty")
            
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("location_id must be a non-empty string")
        
        rows: List[Dict[str, Any]] = []
        
        # Handle different API response structures
        try:
            if 'timeseries' in api_data:
                timeseries = api_data['timeseries']
            elif 'data' in api_data:
                timeseries = api_data['data']
            else:
                timeseries = api_data
        except (TypeError, AttributeError) as e:
            raise ValueError(f"Invalid API data structure: {e}")
        
        if not isinstance(timeseries, list):
            raise ValueError("timeseries data must be a list")
        
        for year_data in timeseries:
            try:
                year = year_data.get('year')
                if year is None:
                    self.logger.warning("Skipping year_data without year field")
                    continue
                    
                # Validate year
                year = int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None
                if year is None or year < 1900 or year > 2100:
                    self.logger.warning(f"Invalid year value: {year_data.get('year')}")
                    continue
                
                season_data = year_data.get('seasons', {})
                if not isinstance(season_data, dict):
                    self.logger.warning(f"Invalid seasons data for year {year}: {type(season_data)}")
                    continue
                
                for season, water_info in season_data.items():
                    # Validate season
                    if season not in VALID_SEASONS:
                        self.logger.warning(f"Unknown season '{season}' for location {location_id}")
                        continue
                    
                    if not isinstance(water_info, dict):
                        self.logger.warning(f"Invalid water_info for season {season}: {type(water_info)}")
                        continue
                    
                    # Extract and validate water area
                    water_area = water_info.get('area_ha', 0)
                    try:
                        water_area = float(water_area)
                        if water_area < MIN_WATER_AREA_HA or water_area > MAX_WATER_AREA_HA:
                            self.logger.warning(
                                f"Water area {water_area} ha out of reasonable range "
                                f"[{MIN_WATER_AREA_HA}, {MAX_WATER_AREA_HA}] for {location_id}"
                            )
                            # Still include but mark as questionable
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid water_area value: {water_area}")
                        water_area = 0.0
                    
                    # Extract and validate water body count
                    water_body_count = water_info.get('count', 0)
                    try:
                        water_body_count = int(water_body_count)
                        if water_body_count < 0:
                            self.logger.warning(f"Negative water body count: {water_body_count}")
                            water_body_count = 0
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid water_body_count value: {water_body_count}")
                        water_body_count = 0
                    
                    row = {
                        'location_id': location_id.strip(),
                        'year': year,
                        'season': season,
                        'water_area_ha': water_area,
                        'water_body_count': water_body_count,
                        'data_quality': water_info.get('quality', 'good')
                    }
                    rows.append(row)
                    
            except Exception as e:
                self.logger.error(f"Error processing year_data: {e}", exc_info=True)
                continue
        
        if not rows:
            self.logger.warning(f"No valid data rows created for location {location_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        self.logger.debug(f"Created DataFrame with {len(df)} rows for location {location_id}")
        return df
    
    def _clean_water_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate water data.
        
        This method performs comprehensive data cleaning including type conversion,
        validation, outlier detection, and data quality checks.
        
        Args:
            df: Raw water data DataFrame with columns: location_id, year, season,
                water_area_ha, water_body_count, data_quality
                
        Returns:
            Cleaned DataFrame with validated data types and removed invalid records
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            
        Example:
            >>> df = pd.DataFrame({
            ...     'location_id': ['V001', 'V002'],
            ...     'year': [2020, 2021],
            ...     'season': ['monsoon', 'winter'],
            ...     'water_area_ha': [100.5, -5.0],  # Invalid negative value
            ...     'water_body_count': [5, 3]
            ... })
            >>> cleaned_df = processor._clean_water_data(df)
            >>> print(len(cleaned_df))
            1  # Only the valid record remains
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = ['location_id', 'year', 'season', 'water_area_ha']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        original_count = len(df)
        self.logger.info(f"Starting data cleaning for {original_count} records")
        
        # Make a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        try:
            # Convert data types with error handling
            df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
            df_clean['water_area_ha'] = pd.to_numeric(df_clean['water_area_ha'], errors='coerce')
            df_clean['water_body_count'] = pd.to_numeric(df_clean['water_body_count'], errors='coerce')
            
            # Validate year range
            invalid_years = df_clean[(df_clean['year'] < 1900) | (df_clean['year'] > 2100)]
            if not invalid_years.empty:
                self.logger.warning(f"Removing {len(invalid_years)} records with invalid years")
                df_clean = df_clean[(df_clean['year'] >= 1900) & (df_clean['year'] <= 2100)]
            
            # Validate seasons
            invalid_seasons = df_clean[~df_clean['season'].isin(VALID_SEASONS)]
            if not invalid_seasons.empty:
                self.logger.warning(f"Removing {len(invalid_seasons)} records with invalid seasons")
                df_clean = df_clean[df_clean['season'].isin(VALID_SEASONS)]
            
            # Remove rows with missing critical data
            before_na_removal = len(df_clean)
            df_clean = df_clean.dropna(subset=['year', 'season', 'water_area_ha'])
            na_removed = before_na_removal - len(df_clean)
            if na_removed > 0:
                self.logger.warning(f"Removed {na_removed} records with missing critical data")
            
            # Remove invalid water areas (negative or unreasonably large)
            before_area_filter = len(df_clean)
            df_clean = df_clean[
                (df_clean['water_area_ha'] >= MIN_WATER_AREA_HA) & 
                (df_clean['water_area_ha'] <= MAX_WATER_AREA_HA)
            ]
            area_filtered = before_area_filter - len(df_clean)
            if area_filtered > 0:
                self.logger.warning(f"Removed {area_filtered} records with invalid water areas")
            
            # Remove negative water body counts
            before_count_filter = len(df_clean)
            df_clean = df_clean[df_clean['water_body_count'] >= 0]
            count_filtered = before_count_filter - len(df_clean)
            if count_filtered > 0:
                self.logger.warning(f"Removed {count_filtered} records with negative water body counts")
            
            # Remove exact duplicates
            before_dedup = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = before_dedup - len(df_clean)
            if duplicates_removed > 0:
                self.logger.warning(f"Removed {duplicates_removed} duplicate records")
            
            # Sort data for consistent ordering
            df_clean = df_clean.sort_values(['location_id', 'year', 'season']).reset_index(drop=True)
            
            # Add data quality flags
            df_clean['data_quality'] = df_clean.get('data_quality', 'good')
            
            # Log summary statistics
            final_count = len(df_clean)
            total_removed = original_count - final_count
            removal_rate = (total_removed / original_count) * 100 if original_count > 0 else 0
            
            self.logger.info(
                f"Data cleaning completed:\n"
                f"- Original records: {original_count}\n"
                f"- Final records: {final_count}\n"
                f"- Records removed: {total_removed} ({removal_rate:.1f}%)\n"
                f"- Unique locations: {df_clean['location_id'].nunique()}\n"
                f"- Year range: {df_clean['year'].min()}-{df_clean['year'].max()}\n"
                f"- Seasons: {sorted(df_clean['season'].unique())}"
            )
            
            # Warn if too much data was removed
            if removal_rate > 50:
                self.logger.warning(
                    f"High data removal rate ({removal_rate:.1f}%). "
                    "Please check data quality and validation rules."
                )
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {e}", exc_info=True)
            raise ValueError(f"Data cleaning failed: {e}")
    
    def load_nrm_impact_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load NRM impact data from CSV file.
        
        This method loads Natural Resource Management (NRM) impact assessment data
        from a CSV file and performs validation and cleaning operations.
        
        Args:
            file_path: Path to CSV file. If None, looks for default location
                      (data_dir/nrm_impact_data.csv). Can be string or Path object.
                      
        Returns:
            DataFrame with cleaned NRM impact data containing columns:
            location_id, year, intervention_type, pond_presence, etc.
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file is empty or contains invalid data
            pd.errors.EmptyDataError: If the CSV file is empty
            pd.errors.ParserError: If the CSV file cannot be parsed
            
        Example:
            >>> processor = DataProcessor('./data')
            >>> df = processor.load_nrm_impact_data('nrm_data.csv')
            >>> print(df.columns.tolist())
            ['location_id', 'year', 'intervention_type', 'pond_presence']
        """
        if file_path is None:
            file_path = self.data_dir / 'nrm_impact_data.csv'
        
        # Convert to Path object for consistent handling
        file_path = Path(file_path)
        
        # Validate file path
        if not file_path.exists():
            raise FileNotFoundError(
                f"NRM impact data file not found: {file_path}. "
                f"Please ensure the file exists and is accessible."
            )
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file size to avoid processing extremely large files
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # Warn for files larger than 100MB
            self.logger.warning(
                f"Large file detected ({file_size_mb:.1f}MB). "
                "Consider processing in chunks or optimizing the data."
            )
        
        self.logger.info(f"Loading NRM impact data from: {file_path}")
        
        try:
            # Load CSV with error handling
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                low_memory=False,  # Prevent mixed type inference warnings
                na_values=['', 'NA', 'N/A', 'null', 'None'],
                keep_default_na=True
            )
            
            if df.empty:
                raise ValueError("CSV file is empty or contains no valid data")
            
            self.logger.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns "
                f"from NRM impact data file"
            )
            
            return self._clean_nrm_data(df)
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file {file_path}: {e}")
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading file {file_path}: {e}")
            # Try with different encoding
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
                self.logger.warning("File read with latin-1 encoding. Consider saving as UTF-8.")
                return self._clean_nrm_data(df)
            except Exception as fallback_error:
                raise ValueError(
                    f"Unable to read file with any encoding. "
                    f"Original error: {e}, Fallback error: {fallback_error}"
                )
        except Exception as e:
            self.logger.error(f"Unexpected error loading NRM impact data: {e}", exc_info=True)
            raise ValueError(f"Failed to load NRM impact data from {file_path}: {e}")
    
    def _clean_nrm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate NRM impact data.
        
        This method performs comprehensive cleaning of Natural Resource Management
        impact data including column standardization, type conversion, validation,
        and data quality checks.
        
        Args:
            df: Raw NRM data DataFrame with columns like location_id, year,
                intervention_type, pond_presence, etc.
                
        Returns:
            Cleaned DataFrame with standardized column names and validated data
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            
        Example:
            >>> df = pd.DataFrame({
            ...     'Location ID': ['V001', 'V002'],
            ...     'Year': [2020, 2021],
            ...     'Pond Presence': ['Yes', 'No'],
            ...     'Intervention Type': ['pond', 'check_dam']
            ... })
            >>> cleaned_df = processor._clean_nrm_data(df)
            >>> print(cleaned_df.columns.tolist())
            ['location_id', 'year', 'pond_presence', 'intervention_type']
        """
        if df.empty:
            raise ValueError("NRM data DataFrame is empty")
        
        original_count = len(df)
        self.logger.info(f"Starting NRM data cleaning for {original_count} records")
        
        # Make a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        try:
            # Standardize column names (lowercase, replace spaces with underscores)
            df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
            self.logger.debug(f"Standardized columns: {df_clean.columns.tolist()}")
            
            # Check for required columns
            required_columns = ['location_id', 'year']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in NRM data: {missing_columns}")
            
            # Convert and validate year
            if 'year' in df_clean.columns:
                before_year_clean = len(df_clean)
                df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
                
                # Validate year range
                invalid_years = df_clean[(df_clean['year'] < 1900) | (df_clean['year'] > 2100)]
                if not invalid_years.empty:
                    self.logger.warning(f"Removing {len(invalid_years)} records with invalid years")
                    df_clean = df_clean[(df_clean['year'] >= 1900) & (df_clean['year'] <= 2100)]
                
                df_clean = df_clean.dropna(subset=['year'])
                year_removed = before_year_clean - len(df_clean)
                if year_removed > 0:
                    self.logger.warning(f"Removed {year_removed} records with invalid year data")
            
            # Clean and validate pond_presence if present
            if 'pond_presence' in df_clean.columns:
                before_pond_clean = len(df_clean)
                
                # Convert to string and standardize
                df_clean['pond_presence'] = df_clean['pond_presence'].astype(str).str.strip().str.lower()
                
                # Map various representations to 0/1
                pond_mapping = {
                    'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1, 'present': 1,
                    'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0, 'absent': 0, 'none': 0
                }
                
                df_clean['pond_presence'] = df_clean['pond_presence'].map(pond_mapping)
                df_clean['pond_presence'] = pd.to_numeric(df_clean['pond_presence'], errors='coerce').fillna(0)
                
                # Ensure only 0 or 1 values
                df_clean['pond_presence'] = df_clean['pond_presence'].clip(0, 1).astype(int)
                
                pond_removed = before_pond_clean - len(df_clean)
                if pond_removed > 0:
                    self.logger.warning(f"Removed {pond_removed} records with invalid pond_presence data")
            
            # Clean intervention_type if present
            if 'intervention_type' in df_clean.columns:
                before_intervention_clean = len(df_clean)
                
                # Standardize intervention types
                df_clean['intervention_type'] = df_clean['intervention_type'].astype(str).str.strip().str.lower()
                
                # Validate intervention types
                invalid_interventions = df_clean[~df_clean['intervention_type'].isin(VALID_INTERVENTION_TYPES + ['none', ''])]
                if not invalid_interventions.empty:
                    self.logger.warning(
                        f"Found {len(invalid_interventions)} records with unrecognized intervention types: "
                        f"{invalid_interventions['intervention_type'].unique()}"
                    )
                    # Keep them but log the issue
                
                intervention_removed = before_intervention_clean - len(df_clean)
                if intervention_removed > 0:
                    self.logger.warning(f"Removed {intervention_removed} records with invalid intervention_type data")
            
            # Remove rows with missing critical data
            before_na_removal = len(df_clean)
            df_clean = df_clean.dropna(subset=required_columns)
            na_removed = before_na_removal - len(df_clean)
            if na_removed > 0:
                self.logger.warning(f"Removed {na_removed} records with missing critical data")
            
            # Remove exact duplicates
            before_dedup = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = before_dedup - len(df_clean)
            if duplicates_removed > 0:
                self.logger.warning(f"Removed {duplicates_removed} duplicate records")
            
            # Sort data for consistent ordering
            df_clean = df_clean.sort_values(['location_id', 'year']).reset_index(drop=True)
            
            # Log summary statistics
            final_count = len(df_clean)
            total_removed = original_count - final_count
            removal_rate = (total_removed / original_count) * 100 if original_count > 0 else 0
            
            self.logger.info(
                f"NRM data cleaning completed:\n"
                f"- Original records: {original_count}\n"
                f"- Final records: {final_count}\n"
                f"- Records removed: {total_removed} ({removal_rate:.1f}%)\n"
                f"- Unique locations: {df_clean['location_id'].nunique()}\n"
                f"- Year range: {df_clean['year'].min()}-{df_clean['year'].max()}"
            )
            
            # Additional data quality info
            if 'pond_presence' in df_clean.columns:
                pond_stats = df_clean['pond_presence'].value_counts().to_dict()
                self.logger.info(f"Pond presence distribution: {pond_stats}")
            
            if 'intervention_type' in df_clean.columns:
                intervention_stats = df_clean['intervention_type'].value_counts().to_dict()
                self.logger.info(f"Intervention types: {intervention_stats}")
            
            # Warn if too much data was removed
            if removal_rate > 50:
                self.logger.warning(
                    f"High NRM data removal rate ({removal_rate:.1f}%). "
                    "Please check data quality and validation rules."
                )
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during NRM data cleaning: {e}", exc_info=True)
            raise ValueError(f"NRM data cleaning failed: {e}")
    
    def merge_datasets(
        self, 
        water_df: pd.DataFrame, 
        nrm_df: pd.DataFrame,
        merge_on: List[str] = ['location_id', 'year']
    ) -> pd.DataFrame:
        """
        Merge water data with NRM impact data.
        
        This method performs a left join between water data and NRM impact data,
        adding indicators for data availability and performing validation.
        
        Args:
            water_df: Water data DataFrame with columns location_id, year, season,
                     water_area_ha, water_body_count, etc.
            nrm_df: NRM impact data DataFrame with columns location_id, year,
                   intervention_type, pond_presence, etc.
            merge_on: List of column names to merge on. Default is ['location_id', 'year'].
                     
        Returns:
            Merged DataFrame containing all water data with matched NRM data.
            Includes 'nrm_data_available' column indicating successful matches.
            
        Raises:
            ValueError: If merge columns don't exist in both DataFrames
            pd.errors.MergeError: If merge operation fails
            
        Example:
            >>> water_df = pd.DataFrame({
            ...     'location_id': ['V001', 'V001'], 'year': [2020, 2021],
            ...     'season': ['monsoon', 'winter'], 'water_area_ha': [100, 80]
            ... })
            >>> nrm_df = pd.DataFrame({
            ...     'location_id': ['V001'], 'year': [2020], 'pond_presence': [1]
            ... })
            >>> merged = processor.merge_datasets(water_df, nrm_df)
            >>> print(merged['nrm_data_available'].tolist())
            [True, False]
        """
        # Input validation
        if water_df.empty:
            raise ValueError("Water data DataFrame cannot be empty")
            
        if nrm_df.empty:
            self.logger.warning("NRM data DataFrame is empty. Merge will result in all NRM fields as NaN.")
        
        if not merge_on:
            raise ValueError("merge_on cannot be empty")
        
        # Ensure merge columns exist in both datasets
        missing_water_cols = [col for col in merge_on if col not in water_df.columns]
        missing_nrm_cols = [col for col in merge_on if col not in nrm_df.columns]
        
        if missing_water_cols:
            raise ValueError(f"Merge columns not found in water data: {missing_water_cols}")
            
        if missing_nrm_cols:
            raise ValueError(f"Merge columns not found in NRM data: {missing_nrm_cols}")
        
        self.logger.info(
            f"Merging datasets on columns: {merge_on}\n"
            f"- Water data: {len(water_df)} records\n"
            f"- NRM data: {len(nrm_df)} records"
        )
        
        try:
            # Check for duplicate keys in NRM data that would cause ambiguous merge
            if len(nrm_df) > 0:
                nrm_duplicates = nrm_df.duplicated(subset=merge_on).sum()
                if nrm_duplicates > 0:
                    self.logger.warning(
                        f"Found {nrm_duplicates} duplicate keys in NRM data. "
                        "This may cause unexpected merge results."
                    )
            
            # Perform left merge (keep all water data, add matching NRM data)
            merged_df = pd.merge(
                water_df, 
                nrm_df, 
                on=merge_on, 
                how='left',
                indicator=True,  # Add merge indicator
                suffixes=('', '_nrm')  # Handle overlapping column names
            )
            
            # Add indicator for NRM data availability
            merged_df['nrm_data_available'] = merged_df['_merge'] == 'both'
            
            # Remove the merge indicator column
            merged_df = merged_df.drop(columns=['_merge'])
            
            # Log merge statistics
            total_records = len(merged_df)
            matched_records = merged_df['nrm_data_available'].sum()
            unmatched_records = total_records - matched_records
            match_rate = (matched_records / total_records) * 100 if total_records > 0 else 0
            
            self.logger.info(
                f"Merge completed:\n"
                f"- Total records: {total_records}\n"
                f"- Records with NRM data: {matched_records} ({match_rate:.1f}%)\n"
                f"- Records without NRM data: {unmatched_records} ({100-match_rate:.1f}%)\n"
                f"- Final columns: {len(merged_df.columns)}"
            )
            
            # Warn if match rate is low
            if match_rate < 50:
                self.logger.warning(
                    f"Low match rate ({match_rate:.1f}%). "
                    "Check if location_id and year values match between datasets."
                )
            
            # Sort for consistent ordering
            merged_df = merged_df.sort_values(merge_on + ['season']).reset_index(drop=True)
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error during dataset merge: {e}", exc_info=True)
            raise ValueError(f"Failed to merge datasets: {e}")
    
    def calculate_water_trends(
        self, 
        df: pd.DataFrame, 
        group_by: List[str] = ['location_id', 'season']
    ) -> pd.DataFrame:
        """
        Calculate water area trends over time.
        
        This method calculates comprehensive trend statistics for water body areas
        grouped by specified dimensions (typically location and season). It includes
        linear trend analysis, descriptive statistics, and data quality indicators.
        
        Args:
            df: Input DataFrame with columns location_id, year, season, water_area_ha
                 and optionally other water-related columns
            group_by: List of column names to group by for trend calculation.
                     Default is ['location_id', 'season'].
                     
        Returns:
            DataFrame with trend statistics including:
            - location_id, season (from group_by)
            - mean_water_area_ha, std_water_area_ha, min_water_area_ha, max_water_area_ha
            - trend_slope_ha_per_year (linear trend slope)
            - data_points, start_year, end_year
            - trend_quality (assessment of trend reliability)
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            
        Example:
            >>> df = pd.DataFrame({
            ...     'location_id': ['V001', 'V001', 'V001'],
            ...     'year': [2020, 2021, 2022],
            ...     'season': ['monsoon', 'monsoon', 'monsoon'],
            ...     'water_area_ha': [100, 110, 105]
            ... })
            >>> trends = processor.calculate_water_trends(df, ['location_id', 'season'])
            >>> print(trends['trend_slope_ha_per_year'].iloc[0])
            2.5  # Positive trend
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for trend calculation")
        
        required_columns = ['water_area_ha', 'year'] + group_by
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for trend calculation: {missing_columns}")
        
        if not group_by:
            raise ValueError("group_by cannot be empty")
        
        self.logger.info(
            f"Calculating water trends for {len(df)} records grouped by: {group_by}"
        )
        
        trend_stats: List[Dict[str, Any]] = []
        total_groups = 0
        successful_calculations = 0
        
        try:
            for group_name, group_df in df.groupby(group_by):
                total_groups += 1
                
                try:
                    # Sort by year for consistent trend calculation
                    group_df = group_df.sort_values('year')
                    
                    # Extract data arrays
                    water_areas = group_df['water_area_ha'].values
                    years = group_df['year'].values
                    
                    # Basic validation
                    if len(water_areas) == 0:
                        self.logger.warning(f"Empty group for {group_name}")
                        continue
                    
                    # Check for valid water area values
                    valid_mask = ~np.isnan(water_areas) & (water_areas >= 0)
                    if not np.any(valid_mask):
                        self.logger.warning(f"No valid water area data for group {group_name}")
                        continue
                    
                    water_areas = water_areas[valid_mask]
                    years = years[valid_mask]
                    
                    # Calculate linear trend (slope) with improved numerical stability
                    slope = 0.0
                    trend_quality = 'insufficient_data'
                    
                    if len(water_areas) >= MIN_DATA_POINTS_FOR_TREND and len(np.unique(years)) > 1:
                        try:
                            # Use robust linear regression to handle outliers
                            if len(water_areas) >= 3:
                                # For sufficient data points, use robust fitting
                                # Scale years to improve numerical stability
                                year_mean = np.mean(years)
                                year_std = np.std(years)
                                
                                if year_std > 0:
                                    years_scaled = (years - year_mean) / year_std
                                    
                                    # Use numpy's polyfit with full output to get diagnostics
                                    coeffs, residuals, rank, singular_values, rcond = np.polyfit(
                                        years_scaled, water_areas, 1, full=True
                                    )
                                    
                                    # Check numerical stability
                                    if rank == 2 and np.all(singular_values > rcond):
                                        slope = coeffs[0] / year_std  # Scale back to original units
                                        trend_quality = 'good'
                                        successful_calculations += 1
                                    else:
                                        self.logger.debug(
                                            f"Poorly conditioned fit for group {group_name}: "
                                            f"rank={rank}, singular_values={singular_values}"
                                        )
                                        trend_quality = 'poor_fit'
                                        # Fall back to simple linear regression
                                        slope = self._simple_linear_regression(years, water_areas)
                                else:
                                    # All years are the same, cannot calculate trend
                                    trend_quality = 'constant_year'
                            else:
                                # For minimal data points, use simple calculation
                                if len(water_areas) == 2:
                                    year_diff = years[1] - years[0]
                                    if year_diff != 0:
                                        slope = (water_areas[1] - water_areas[0]) / year_diff
                                        trend_quality = 'minimal_data'
                                        successful_calculations += 1
                        except (np.RankWarning, np.linalg.LinAlgError, ValueError) as e:
                            self.logger.debug(f"Trend calculation failed for group {group_name}: {e}")
                            # Fall back to simple method
                            slope = self._simple_linear_regression(years, water_areas)
                            trend_quality = 'fallback'
                            if slope != 0:
                                successful_calculations += 1
                    
                    # Calculate comprehensive statistics
                    stats = {
                        # Group identifiers
                        **{group_by[i]: group_name[i] for i in range(len(group_by))},
                        
                        # Basic statistics
                        'mean_water_area_ha': float(np.mean(water_areas)),
                        'std_water_area_ha': float(np.std(water_areas, ddof=1)),
                        'min_water_area_ha': float(np.min(water_areas)),
                        'max_water_area_ha': float(np.max(water_areas)),
                        'median_water_area_ha': float(np.median(water_areas)),
                        
                        # Trend information
                        'trend_slope_ha_per_year': float(slope),
                        'trend_quality': trend_quality,
                        
                        # Data coverage
                        'data_points': len(water_areas),
                        'start_year': int(np.min(years)),
                        'end_year': int(np.max(years)),
                        'year_span': int(np.max(years) - np.min(years)),
                        
                        # Additional metrics
                        'coefficient_of_variation': float(np.std(water_areas, ddof=1) / np.mean(water_areas)) if np.mean(water_areas) > 0 else 0.0,
                        'total_observations': len(group_df),  # Original group size
                    }
                    
                    trend_stats.append(stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing group {group_name}: {e}", exc_info=True)
                    continue
            
            if not trend_stats:
                raise ValueError("No valid trend statistics could be calculated")
            
            # Create result DataFrame
            result_df = pd.DataFrame(trend_stats)
            
            # Sort results for consistent ordering
            result_df = result_df.sort_values(group_by).reset_index(drop=True)
            
            # Log summary statistics
            self.logger.info(
                f"Trend calculation completed:\n"
                f"- Total groups processed: {total_groups}\n"
                f"- Successful trend calculations: {successful_calculations}\n"
                f"- Success rate: {(successful_calculations/total_groups)*100:.1f}%\n"
                f"- Average trend slope: {result_df['trend_slope_ha_per_year'].mean():.3f} ha/year"
            )
            
            # Trend quality distribution
            quality_dist = result_df['trend_quality'].value_counts().to_dict()
            self.logger.info(f"Trend quality distribution: {quality_dist}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error during trend calculation: {e}", exc_info=True)
            raise ValueError(f"Failed to calculate water trends: {e}")
    
    def _simple_linear_regression(self, years: np.ndarray, water_areas: np.ndarray) -> float:
        """
        Calculate simple linear regression slope as fallback method.
        
        This method provides a basic slope calculation when the main
        polynomial fitting fails due to numerical issues.
        
        Args:
            years: Array of years
            water_areas: Array of water areas
            
        Returns:
            Linear regression slope (water area change per year)
        """
        try:
            if len(years) < 2:
                return 0.0
            
            # Calculate covariance and variance
            year_mean = np.mean(years)
            area_mean = np.mean(water_areas)
            
            numerator = np.sum((years - year_mean) * (water_areas - area_mean))
            denominator = np.sum((years - year_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            return float(numerator / denominator)
            
        except Exception:
            return 0.0
    
    def aggregate_by_intervention(
        self, 
        df: pd.DataFrame, 
        intervention_col: str = 'pond_presence'
    ) -> pd.DataFrame:
        """
        Aggregate water data by intervention presence.
        
        This method calculates comprehensive statistics for water data grouped
        by intervention presence (e.g., ponds vs no ponds) to assess the impact
        of NRM interventions on water body characteristics.
        
        Args:
            df: Input DataFrame with water data and intervention indicators
            intervention_col: Column name indicating intervention presence.
                           Should contain binary values (0/1) or boolean.
                           Default is 'pond_presence'.
                           
        Returns:
            Aggregated DataFrame with statistics for each intervention group:
            - intervention_col, intervention_type (labels)
            - water_area_ha statistics (mean, std, min, max, count)
            - water_body_count statistics (mean, std)
            - location_id_nunique (number of unique locations)
            - Additional derived metrics
            
        Raises:
            ValueError: If intervention column doesn't exist or contains invalid data
            
        Example:
            >>> df = pd.DataFrame({
            ...     'location_id': ['V001', 'V002', 'V003', 'V004'],
            ...     'water_area_ha': [100, 80, 120, 90],
            ...     'pond_presence': [1, 0, 1, 0]
            ... })
            >>> agg = processor.aggregate_by_intervention(df, 'pond_presence')
            >>> print(agg[['intervention_type', 'water_area_ha_mean']].values.tolist())
            [['With Intervention', 110.0], ['No Intervention', 85.0]]
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for intervention aggregation")
        
        if intervention_col not in df.columns:
            available_cols = df.columns.tolist()
            raise ValueError(
                f"Intervention column '{intervention_col}' not found. "
                f"Available columns: {available_cols}"
            )
        
        required_columns = ['water_area_ha', 'location_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info(
            f"Aggregating data by intervention column: {intervention_col}"
        )
        
        try:
            # Validate and clean intervention column
            df_clean = df.copy()
            
            # Convert intervention column to numeric if needed
            if df_clean[intervention_col].dtype == 'object':
                # Handle string representations
                df_clean[intervention_col] = df_clean[intervention_col].astype(str).str.lower()
                mapping = {'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1, 'present': 1,
                          'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0, 'absent': 0, 'none': 0}
                df_clean[intervention_col] = df_clean[intervention_col].map(mapping)
            
            # Convert to numeric and handle missing values
            df_clean[intervention_col] = pd.to_numeric(
                df_clean[intervention_col], errors='coerce'
            ).fillna(0)
            
            # Ensure only 0 or 1 values
            df_clean[intervention_col] = df_clean[intervention_col].clip(0, 1).astype(int)
            
            # Check for valid water area data
            valid_mask = df_clean['water_area_ha'] >= 0
            if not valid_mask.all():
                self.logger.warning(
                    f"Found {(~valid_mask).sum()} records with negative water areas. "
                    "These will be excluded from aggregation."
                )
                df_clean = df_clean[valid_mask]
            
            if df_clean.empty:
                raise ValueError("No valid data remaining after filtering")
            
            # Group by intervention presence and calculate comprehensive statistics
            agg_dict = {
                'water_area_ha': [
                    'mean', 'std', 'min', 'max', 'count', 'median',
                    lambda x: (x >= 0).sum()  # Count of non-negative values
                ],
                'water_body_count': ['mean', 'std', 'min', 'max', 'sum'] if 'water_body_count' in df_clean.columns else ['mean'],
                'location_id': 'nunique'
            }
            
            # Add water_body_count only if it exists
            if 'water_body_count' in df_clean.columns:
                agg_dict['water_body_count'] = ['mean', 'std', 'min', 'max', 'sum']
            
            # Perform aggregation
            agg_stats = df_clean.groupby(intervention_col).agg(agg_dict)
            
            # Flatten column names and handle custom functions
            new_columns = []
            for col in agg_stats.columns:
                if isinstance(col[1], str):
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    # Handle lambda function
                    new_columns.append(f"{col[0]}_valid_count")
            
            agg_stats.columns = new_columns
            agg_stats = agg_stats.reset_index()
            
            # Add intervention labels
            intervention_labels = {
                0: 'No Intervention',
                1: 'With Intervention'
            }
            
            agg_stats['intervention_type'] = agg_stats[intervention_col].map(intervention_labels)
            
            # Calculate additional derived metrics
            if 'water_area_ha_mean' in agg_stats.columns and 'water_area_ha_std' in agg_stats.columns:
                # Coefficient of variation
                agg_stats['water_area_ha_cv'] = (
                    agg_stats['water_area_ha_std'] / agg_stats['water_area_ha_mean']
                ).fillna(0)
            
            # Calculate percentage difference if both groups exist
            if len(agg_stats) == 2:
                with_intervention = agg_stats[agg_stats[intervention_col] == 1]
                without_intervention = agg_stats[agg_stats[intervention_col] == 0]
                
                if not with_intervention.empty and not without_intervention.empty:
                    with_mean = with_intervention['water_area_ha_mean'].iloc[0]
                    without_mean = without_intervention['water_area_ha_mean'].iloc[0]
                    
                    if without_mean > 0:
                        percent_increase = ((with_mean - without_mean) / without_mean) * 100
                        agg_stats['water_area_increase_pct'] = agg_stats[intervention_col].map({
                            1: percent_increase,
                            0: 0.0
                        })
            
            # Sort by intervention presence for consistent ordering
            agg_stats = agg_stats.sort_values(intervention_col).reset_index(drop=True)
            
            # Log summary statistics
            self.logger.info(
                f"Intervention aggregation completed:\n"
                f"- Groups analyzed: {len(agg_stats)}\n"
                f"- Total records: {len(df_clean)}\n"
                f"- Unique locations: {agg_stats['location_id_nunique'].sum()}"
            )
            
            # Log group-specific information
            for _, row in agg_stats.iterrows():
                group_name = row['intervention_type']
                locations = row['location_id_nunique']
                mean_area = row.get('water_area_ha_mean', 'N/A')
                self.logger.info(
                    f"{group_name}: {locations} locations, "
                    f"mean water area: {mean_area} ha"
                )
            
            return agg_stats
            
        except Exception as e:
            self.logger.error(f"Error during intervention aggregation: {e}", exc_info=True)
            raise ValueError(f"Failed to aggregate by intervention: {e}")
    
    def create_seasonal_summary(
        self, 
        df: pd.DataFrame, 
        location_level: str = 'location_id'
    ) -> pd.DataFrame:
        """
        Create seasonal summary statistics.
        
        This method generates comprehensive seasonal summaries of water data,
        aggregated by specified spatial level (typically location_id). It includes
        descriptive statistics, data quality indicators, and temporal coverage.
        
        Args:
            df: Input DataFrame with water data including location_id, season,
                water_area_ha, water_body_count, year, etc.
            location_level: Column name for spatial aggregation level.
                          Default is 'location_id'. Can be any geographic identifier.
                          
        Returns:
            Seasonal summary DataFrame with columns:
            - location_level, season (grouping columns)
            - water_area_ha statistics (mean, std, min, max, median, count)
            - water_body_count statistics (mean, std, min, max, sum)
            - Temporal coverage (year_min, year_max, year_count, year_span)
            - Data quality indicators (data_completeness, coefficient_of_variation)
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            
        Example:
            >>> df = pd.DataFrame({
            ...     'location_id': ['V001', 'V001', 'V001', 'V002'],
            ...     'season': ['monsoon', 'winter', 'monsoon', 'monsoon'],
            ...     'water_area_ha': [100, 80, 110, 90],
            ...     'year': [2020, 2020, 2021, 2020]
            ... })
            >>> summary = processor.create_seasonal_summary(df)
            >>> print(summary[['location_id', 'season', 'water_area_ha_mean']].head())
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for seasonal summary")
        
        required_columns = [location_level, 'season', 'water_area_ha', 'year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info(
            f"Creating seasonal summary for {len(df)} records at level: {location_level}"
        )
        
        try:
            # Filter for valid data
            df_clean = df.copy()
            valid_mask = df_clean['water_area_ha'] >= 0
            df_clean = df_clean[valid_mask]
            
            if df_clean.empty:
                raise ValueError("No valid data remaining after filtering")
            
            # Group by location and season for comprehensive statistics
            agg_dict = {
                'water_area_ha': ['mean', 'std', 'min', 'max', 'median', 'count'],
                'year': ['min', 'max', 'nunique'],
                'location_id': 'count'  # Total observations per group
            }
            
            # Add water_body_count statistics if available
            if 'water_body_count' in df_clean.columns:
                agg_dict['water_body_count'] = ['mean', 'std', 'min', 'max', 'sum']
            
            # Perform aggregation
            seasonal_summary = df_clean.groupby([location_level, 'season']).agg(agg_dict)
            
            # Flatten column names
            seasonal_summary.columns = ['_'.join(col).strip() for col in seasonal_summary.columns]
            seasonal_summary = seasonal_summary.reset_index()
            
            # Calculate derived metrics
            # Year span
            seasonal_summary['year_span'] = seasonal_summary['year_max'] - seasonal_summary['year_min']
            
            # Coefficient of variation for water area
            seasonal_summary['water_area_ha_cv'] = (
                seasonal_summary['water_area_ha_std'] / seasonal_summary['water_area_ha_mean']
            ).fillna(0)
            
            # Data completeness (years with data / total possible years)
            seasonal_summary['data_completeness'] = (
                seasonal_summary['year_nunique'] / seasonal_summary['year_span']
            ).fillna(0)
            
            # Add data quality flags
            seasonal_summary['data_quality'] = 'good'  # Default
            
            # Flag groups with limited data
            seasonal_summary.loc[seasonal_summary['water_area_ha_count'] < 3, 'data_quality'] = 'limited_data'
            seasonal_summary.loc[seasonal_summary['water_area_ha_cv'] > 1.0, 'data_quality'] = 'high_variability'
            seasonal_summary.loc[seasonal_summary['data_completeness'] < 0.5, 'data_quality'] = 'gaps_in_data'
            
            # Sort for consistent ordering
            seasonal_summary = seasonal_summary.sort_values([location_level, 'season']).reset_index(drop=True)
            
            # Log summary statistics
            total_groups = len(seasonal_summary)
            unique_locations = seasonal_summary[location_level].nunique()
            seasons_covered = seasonal_summary['season'].unique()
            
            self.logger.info(
                f"Seasonal summary completed:\n"
                f"- Total groups: {total_groups}\n"
                f"- Unique {location_level}s: {unique_locations}\n"
                f"- Seasons covered: {sorted(seasons_covered)}\n"
                f"- Year range: {seasonal_summary['year_min'].min()}-{seasonal_summary['year_max'].max()}"
            )
            
            # Data quality distribution
            quality_dist = seasonal_summary['data_quality'].value_counts().to_dict()
            self.logger.info(f"Data quality distribution: {quality_dist}")
            
            return seasonal_summary
            
        except Exception as e:
            self.logger.error(f"Error creating seasonal summary: {e}", exc_info=True)
            raise ValueError(f"Failed to create seasonal summary: {e}")
    
    def export_processed_data(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        format: str = 'csv'
    ) -> None:
        """
        Export processed data to file.
        
        This method exports processed DataFrames to various formats with proper
        validation, error handling, and logging. It creates the output directory
        if needed and provides feedback on the export operation.
        
        Args:
            df: DataFrame to export
            filename: Output filename without extension
            format: Export format. Supported formats: 'csv', 'excel', 'parquet'.
                   Default is 'csv'.
                   
        Raises:
            ValueError: If DataFrame is empty, filename is invalid, or format unsupported
            OSError: If unable to create output directory or write file
            
        Example:
            >>> processor = DataProcessor('./data')
            >>> df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            >>> processor.export_processed_data(df, 'my_data', 'csv')
            # Creates: ./data/processed/my_data.csv
        """
        # Input validation
        if df.empty:
            raise ValueError("Cannot export empty DataFrame")
        
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string")
        
        # Sanitize filename
        filename = filename.strip()
        if not filename.replace('_', '').replace('-', '').replace('.', '').isalnum():
            self.logger.warning(f"Filename '{filename}' contains special characters")
        
        supported_formats = ['csv', 'excel', 'parquet']
        if format not in supported_formats:
            raise ValueError(
                f"Unsupported export format: {format}. "
                f"Supported formats: {supported_formats}"
            )
        
        # Create output path
        output_path = self.data_dir / 'processed' / filename
        
        try:
            # Create processed directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists
            final_path = output_path.with_suffix(self._get_file_extension(format))
            if final_path.exists():
                self.logger.warning(f"Overwriting existing file: {final_path}")
            
            # Export based on format
            if format == 'csv':
                df.to_csv(final_path, index=False, encoding='utf-8')
            elif format == 'excel':
                try:
                    df.to_excel(final_path, index=False, engine='openpyxl')
                except ImportError:
                    # Fallback to xlsxwriter if openpyxl not available
                    df.to_excel(final_path, index=False, engine='xlsxwriter')
            elif format == 'parquet':
                try:
                    df.to_parquet(final_path, index=False, engine='pyarrow')
                except ImportError:
                    # Fallback to fastparquet if pyarrow not available
                    df.to_parquet(final_path, index=False, engine='fastparquet')
            
            # Get file size for logging
            file_size_mb = final_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(
                f"Successfully exported {len(df)} rows and {len(df.columns)} columns "
                f"to {final_path} ({file_size_mb:.2f} MB)"
            )
            
        except PermissionError:
            raise OSError(f"Permission denied when writing to {final_path}")
        except OSError as e:
            raise OSError(f"Failed to create output directory or write file: {e}")
        except Exception as e:
            self.logger.error(f"Error during data export: {e}", exc_info=True)
            raise ValueError(f"Failed to export data: {e}")
    
    def _get_file_extension(self, format: str) -> str:
        """
        Get file extension for export format.
        
        Args:
            format: Export format name
            
        Returns:
            File extension including dot
        """
        extensions = {
            'csv': '.csv',
            'excel': '.xlsx',
            'parquet': '.parquet'
        }
        return extensions.get(format, '.csv')


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
