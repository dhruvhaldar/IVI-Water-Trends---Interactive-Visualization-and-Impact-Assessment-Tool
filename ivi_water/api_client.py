"""
CoRE Stack API Client Module

This module provides functions to interact with CoRE Stack REST APIs
for fetching spatial units and seasonal surface water data.
"""

# Standard library imports
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout, ConnectionError
from urllib3.util.retry import Retry

# Constants
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
REQUEST_TIMEOUT = 30  # seconds
DEFAULT_BASE_URL = 'https://api.corestack.org/v1'
USER_AGENT = 'IVI-Water-Trends/0.1.0'

# Logger setup
logger = logging.getLogger(__name__)


class CoREStackClient:
    """
    Client for interacting with CoRE Stack APIs.
    
    This class provides a robust interface for communicating with CoRE Stack
    REST APIs, including authentication, caching, rate limiting, and comprehensive
    error handling for fetching spatial units and seasonal surface water data.
    
    Attributes:
        api_key (str): API key for authentication
        base_url (str): Base URL for the API endpoints
        session (requests.Session): HTTP session with retry strategy
        cache_ttl (int): Cache time-to-live in seconds
        logger (logging.Logger): Logger instance for this class
        
    Example:
        >>> client = CoREStackClient(api_key="your-api-key")
        >>> locations = client.get_spatial_units()
        >>> water_data = client.get_seasonal_water_data("V001", 2020, 2022)
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """
        Initialize the CoRE Stack API client.
        
        This method sets up the HTTP session with retry strategy, authentication
        headers, and caching mechanism for efficient API communication.
        
        Args:
            api_key: API key for authentication. If None, reads from CORE_API_KEY
                    environment variable.
            base_url: Base URL for the API. If None, uses CORE_API_BASE_URL
                    environment variable or default CoRE Stack URL.
                    
        Raises:
            ValueError: If API key is not provided or found in environment
            
        Example:
            >>> # Using environment variables
            >>> client = CoREStackClient()
            >>> # Using explicit parameters
            >>> client = CoREStackClient(
            ...     api_key="your-key",
            ...     base_url="https://custom.api.example.com/v1"
            ... )
        """
        self.api_key = api_key or os.getenv('CORE_API_KEY')
        self.base_url = base_url or os.getenv('CORE_API_BASE_URL', DEFAULT_BASE_URL)
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key is required. Set CORE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        if not isinstance(self.api_key, str) or len(self.api_key.strip()) == 0:
            raise ValueError("API key must be a non-empty string")
        
        # Validate base URL
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("Base URL must be a non-empty string")
        
        self.base_url = self.base_url.rstrip('/')  # Remove trailing slash
        
        self.logger = logging.getLogger(__name__)
        
        # Setup session with robust retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key.strip()}',
            'Content-Type': 'application/json',
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        })
        
        # Simple in-memory cache with validation
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = int(os.getenv('CACHE_TTL', str(DEFAULT_CACHE_TTL)))
        
        if self.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")
        
        self.logger.info(
            f"Initialized CoRE Stack client with base URL: {self.base_url}, "
            f"cache TTL: {self.cache_ttl}s"
        )
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        This method validates cache entries based on their timestamp and TTL.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            True if cache entry exists and is within TTL, False otherwise
        """
        if cache_key not in self._cache_timestamps:
            return False
        
        try:
            cache_time = self._cache_timestamps[cache_key]
            age = datetime.now() - cache_time
            return age < timedelta(seconds=self.cache_ttl)
        except Exception as e:
            self.logger.warning(f"Error checking cache validity for {cache_key}: {e}")
            return False
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if valid.
        
        This method retrieves cached data if it exists and hasn't expired.
        Invalid or expired entries are automatically cleaned up.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            Cached data dictionary if valid, None otherwise
        """
        try:
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"Cache hit for {cache_key}")
                return self._cache[cache_key].copy()  # Return a copy to prevent mutation
            else:
                # Clean up expired entry
                self._cleanup_cache_entry(cache_key)
                return None
        except Exception as e:
            self.logger.warning(f"Error retrieving from cache for {cache_key}: {e}")
            return None
    
    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Set data in cache.
        
        This method stores data in the in-memory cache with a timestamp.
        The cache automatically manages size by removing oldest entries when needed.
        
        Args:
            cache_key: Unique identifier for the cached data
            data: Data dictionary to cache
        """
        try:
            if not isinstance(data, dict):
                self.logger.warning(f"Cannot cache non-dict data for {cache_key}")
                return
            
            # Simple cache size management - limit to 1000 entries
            if len(self._cache) >= 1000:
                self._cleanup_oldest_cache_entries()
            
            self._cache[cache_key] = data.copy()  # Store a copy
            self._cache_timestamps[cache_key] = datetime.now()
            self.logger.debug(f"Cached data for {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error setting cache for {cache_key}: {e}")
    
    def _cleanup_cache_entry(self, cache_key: str) -> None:
        """
        Remove a specific cache entry.
        
        Args:
            cache_key: Cache key to remove
        """
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)
    
    def _cleanup_oldest_cache_entries(self, count: int = 100) -> None:
        """
        Remove oldest cache entries to free up space.
        
        Args:
            count: Number of oldest entries to remove
        """
        if not self._cache_timestamps:
            return
        
        # Sort by timestamp and remove oldest
        sorted_entries = sorted(
            self._cache_timestamps.items(), 
            key=lambda x: x[1]
        )
        
        for cache_key, _ in sorted_entries[:count]:
            self._cleanup_cache_entry(cache_key)
        
        self.logger.debug(f"Cleaned up {count} oldest cache entries")
    
    def clear_cache(self) -> None:
        """
        Clear all cached data.
        
        This method removes all entries from the cache, useful for testing
        or when fresh data is required.
        """
        cache_size = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info(f"Cleared {cache_size} cache entries")
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        use_cache: bool = True,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """
        Make a request to the CoRE Stack API.
        
        This method handles HTTP requests with comprehensive error handling,
        caching, retry logic, and response validation.
        
        Args:
            endpoint: API endpoint path (without leading slash)
            params: Query parameters dictionary. Keys and values will be validated.
            use_cache: Whether to use caching for this request. Default is True.
            method: HTTP method to use. Default is 'GET'.
            
        Returns:
            JSON response data from the API
            
        Raises:
            ValueError: If endpoint or parameters are invalid
            RequestException: If the HTTP request fails after retries
            Timeout: If the request times out
            ConnectionError: If connection to API fails
            json.JSONDecodeError: If response is not valid JSON
            
        Example:
            >>> client = CoREStackClient()
            >>> data = client._make_request('spatial-units', {'type': 'village'})
            >>> print(data.get('units', []))
        """
        # Input validation
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("Endpoint must be a non-empty string")
        
        endpoint = endpoint.lstrip('/')  # Remove leading slash
        
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Validate parameter keys and values
        for key, value in params.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"Invalid parameter key: {key}")
            
            # Convert non-string values to strings for URL parameters
            if not isinstance(value, str):
                params[key] = str(value)
        
        # Construct full URL
        url = f"{self.base_url}/{endpoint}"
        
        # Create cache key
        try:
            cache_key = f"{method}_{url}_{json.dumps(params, sort_keys=True)}"
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Cannot create cache key due to non-serializable params: {e}")
            cache_key = f"{method}_{url}_{hash(str(params))}"
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Returning cached data for {endpoint}")
                return cached_data
        
        # Make HTTP request
        try:
            self.logger.debug(f"Making {method} request to {url} with params: {params}")
            
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params if method.upper() == 'GET' else None,
                json=params if method.upper() == 'POST' else None,
                timeout=REQUEST_TIMEOUT
            )
            
            # Handle different response statuses
            if response.status_code == 401:
                raise RequestException("Authentication failed. Check your API key.")
            elif response.status_code == 403:
                raise RequestException("Access forbidden. Insufficient permissions.")
            elif response.status_code == 404:
                raise RequestException(f"Endpoint not found: {endpoint}")
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After', 'unknown')
                raise RequestException(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            elif response.status_code >= 500:
                raise RequestException(f"Server error: {response.status_code}")
            
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response from {url}: {e}")
                self.logger.debug(f"Response content: {response.text[:500]}")
                raise json.JSONDecodeError(f"Invalid JSON response from API: {e}", e.doc, e.pos)
            
            # Validate response structure
            if not isinstance(data, dict):
                self.logger.warning(f"Unexpected response format from {url}: expected dict, got {type(data)}")
                # Still return the data but log the issue
            
            # Cache successful response
            if use_cache:
                self._set_cache(cache_key, data)
            
            self.logger.debug(f"Successfully received response from {url}")
            return data
            
        except Timeout:
            self.logger.error(f"Request timeout for {url} after {REQUEST_TIMEOUT}s")
            raise Timeout(f"Request timeout for {url}")
        except ConnectionError as e:
            self.logger.error(f"Connection error for {url}: {e}")
            raise ConnectionError(f"Failed to connect to {url}: {e}")
        except RequestException:
            # Re-raise RequestException as-is (already logged)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during request to {url}: {e}", exc_info=True)
            raise RequestException(f"Unexpected error during API request: {e}")
    
    def get_spatial_units(self, unit_type: str = "village", state: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available spatial units from CoRE Stack.
        
        This method retrieves spatial units (villages, micro-watersheds, tehsils)
        from the CoRE Stack API with optional filtering by type and state.
        
        Args:
            unit_type: Type of spatial unit. Valid options are 'village', 
                      'micro-watershed', 'tehsil'. Default is 'village'.
            state: Optional state filter to limit results to specific states.
                   Must be a valid state name or code.
                   
        Returns:
            List of spatial unit information dictionaries. Each dictionary contains
            keys like 'id', 'name', 'type', 'state', and other metadata.
            Returns empty list if no units found.
            
        Raises:
            ValueError: If unit_type is invalid or state is malformed
            RequestException: If API request fails
            
        Example:
            >>> client = CoREStackClient()
            >>> villages = client.get_spatial_units('village', 'Maharashtra')
            >>> print(f"Found {len(villages)} villages")
            >>> if villages:
            ...     print(villages[0]['name'])
        """
        # Input validation
        if not isinstance(unit_type, str) or not unit_type.strip():
            raise ValueError("unit_type must be a non-empty string")
        
        valid_unit_types = ['village', 'micro-watershed', 'tehsil', 'watershed', 'block', 'district']
        unit_type = unit_type.strip().lower()
        
        if unit_type not in valid_unit_types:
            raise ValueError(
                f"Invalid unit_type '{unit_type}'. Valid options: {valid_unit_types}"
            )
        
        # Build request parameters
        params: Dict[str, str] = {"type": unit_type}
        
        if state is not None:
            if not isinstance(state, str) or not state.strip():
                raise ValueError("state must be a non-empty string if provided")
            
            params["state"] = state.strip()
        
        self.logger.info(
            f"Fetching spatial units of type '{unit_type}'"
            f"{' for state: ' + state if state else ''}"
        )
        
        try:
            response = self._make_request("spatial-units", params)
            
            # Extract and validate data
            data = response.get("data", [])
            
            if not isinstance(data, list):
                self.logger.warning(f"Expected list from spatial-units endpoint, got {type(data)}")
                return []
            
            # Validate each unit entry
            valid_units = []
            for unit in data:
                if isinstance(unit, dict) and 'id' in unit and 'name' in unit:
                    valid_units.append(unit)
                else:
                    self.logger.warning(f"Invalid spatial unit entry: {unit}")
            
            self.logger.info(f"Retrieved {len(valid_units)} valid spatial units")
            return valid_units
            
        except Exception as e:
            self.logger.error(f"Failed to fetch spatial units: {e}", exc_info=True)
            raise
    
    def get_seasonal_water_data(
        self, 
        location_id: str, 
        start_year: int, 
        end_year: int,
        seasons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get seasonal surface water data for a specific location.
        
        This method retrieves seasonal surface water area measurements for a specific
        spatial unit over a specified time period, with optional season filtering.
        
        Args:
            location_id: Spatial unit identifier (e.g., village code, watershed ID)
            start_year: Start year for data collection (inclusive)
            end_year: End year for data collection (inclusive)
            seasons: List of seasons to include. Valid options are 'perennial', 
                    'winter', 'monsoon', 'summer'. If None, gets all seasons.
                    
        Returns:
            Dictionary containing seasonal water area time series data.
            Structure includes 'timeseries' list with yearly data and metadata.
            Returns empty dict if no data found.
            
        Raises:
            ValueError: If location_id is invalid, years are out of range, or seasons are invalid
            RequestException: If API request fails
            
        Example:
            >>> client = CoREStackClient()
            >>> data = client.get_seasonal_water_data(
            ...     'V001', 2020, 2022, ['monsoon', 'winter']
            ... )
            >>> print(f"Found data for {len(data.get('timeseries', []))} years")
        """
        # Input validation
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("location_id must be a non-empty string")
        
        location_id = location_id.strip()
        
        # Validate years
        current_year = datetime.now().year
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            raise ValueError("start_year and end_year must be integers")
        
        if start_year < 1900 or start_year > current_year + 5:
            raise ValueError(f"start_year {start_year} is out of reasonable range")
        
        if end_year < 1900 or end_year > current_year + 5:
            raise ValueError(f"end_year {end_year} is out of reasonable range")
        
        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        
        # Validate seasons
        valid_seasons = ['perennial', 'winter', 'monsoon', 'summer']
        if seasons is not None:
            if not isinstance(seasons, list):
                raise ValueError("seasons must be a list if provided")
            
            seasons = [s.strip().lower() for s in seasons]
            invalid_seasons = [s for s in seasons if s not in valid_seasons]
            
            if invalid_seasons:
                raise ValueError(
                    f"Invalid seasons: {invalid_seasons}. "
                    f"Valid options: {valid_seasons}"
                )
        
        # Build request parameters
        params: Dict[str, Union[str, int]] = {
            "location_id": location_id,
            "start_year": start_year,
            "end_year": end_year
        }
        
        if seasons:
            params["seasons"] = ",".join(seasons)
        
        self.logger.info(
            f"Fetching seasonal water data for location '{location_id}' "
            f"from {start_year} to {end_year}"
            f"{' for seasons: ' + ', '.join(seasons) if seasons else ''}"
        )
        
        try:
            response = self._make_request("water-trends/seasonal", params)
            
            # Extract and validate data
            data = response.get("data", {})
            
            if not isinstance(data, dict):
                self.logger.warning(f"Expected dict from water-trends/seasonal endpoint, got {type(data)}")
                return {}
            
            # Validate timeseries data if present
            if 'timeseries' in data:
                timeseries = data['timeseries']
                if not isinstance(timeseries, list):
                    self.logger.warning(f"Expected list for timeseries, got {type(timeseries)}")
                    data['timeseries'] = []
                else:
                    # Validate each timeseries entry
                    valid_entries = []
                    for entry in timeseries:
                        if isinstance(entry, dict) and 'year' in entry:
                            valid_entries.append(entry)
                        else:
                            self.logger.warning(f"Invalid timeseries entry: {entry}")
                    
                    data['timeseries'] = valid_entries
                    
                    self.logger.info(
                        f"Retrieved water data with {len(valid_entries)} yearly entries"
                    )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch seasonal water data: {e}", exc_info=True)
            raise
    
    def get_water_trends_summary(
        self, 
        location_ids: List[str], 
        start_year: int, 
        end_year: int
    ) -> Dict[str, Any]:
        """
        Get summary statistics for multiple locations.
        
        This method retrieves aggregated water trends statistics across multiple
        spatial units, providing comparative analysis and summary metrics.
        
        Args:
            location_ids: List of spatial unit identifiers to analyze
            start_year: Start year for data analysis (inclusive)
            end_year: End year for data analysis (inclusive)
            
        Returns:
            Dictionary containing summary statistics including:
            - location_summaries: Per-location statistics
            - aggregated_metrics: Combined statistics across all locations
            - temporal_trends: Year-over-year changes
            Returns empty dict if no data found.
            
        Raises:
            ValueError: If location_ids is empty or invalid, or years are out of range
            RequestException: If API request fails
            
        Example:
            >>> client = CoREStackClient()
            >>> summary = client.get_water_trends_summary(
            ...     ['V001', 'V002', 'V003'], 2020, 2022
            ... )
            >>> print(f"Analyzed {len(summary.get('location_summaries', {}))} locations")
        """
        # Input validation
        if not isinstance(location_ids, list) or not location_ids:
            raise ValueError("location_ids must be a non-empty list")
        
        # Validate each location ID
        valid_location_ids = []
        for loc_id in location_ids:
            if isinstance(loc_id, str) and loc_id.strip():
                valid_location_ids.append(loc_id.strip())
            else:
                self.logger.warning(f"Skipping invalid location ID: {loc_id}")
        
        if not valid_location_ids:
            raise ValueError("No valid location IDs provided")
        
        # Validate years (reuse logic from seasonal_water_data)
        current_year = datetime.now().year
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            raise ValueError("start_year and end_year must be integers")
        
        if start_year < 1900 or start_year > current_year + 5:
            raise ValueError(f"start_year {start_year} is out of reasonable range")
        
        if end_year < 1900 or end_year > current_year + 5:
            raise ValueError(f"end_year {end_year} is out of reasonable range")
        
        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")
        
        # Build request parameters
        params: Dict[str, Union[str, int]] = {
            "location_ids": ",".join(valid_location_ids),
            "start_year": start_year,
            "end_year": end_year
        }
        
        self.logger.info(
            f"Fetching water trends summary for {len(valid_location_ids)} locations "
            f"from {start_year} to {end_year}"
        )
        
        try:
            response = self._make_request("water-trends/summary", params)
            
            # Extract and validate data
            data = response.get("data", {})
            
            if not isinstance(data, dict):
                self.logger.warning(f"Expected dict from water-trends/summary endpoint, got {type(data)}")
                return {}
            
            self.logger.info(f"Retrieved summary statistics for {len(valid_location_ids)} locations")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch water trends summary: {e}", exc_info=True)
            raise
    
    def get_elevation_data(self, location_id: str) -> Dict[str, Any]:
        """
        Get elevation data for a location (for 3D visualization).
        
        This method retrieves elevation and terrain data for a specific spatial unit,
        useful for 3D visualization and topographic analysis.
        
        Args:
            location_id: Spatial unit identifier (e.g., village code, watershed ID)
            
        Returns:
            Dictionary containing elevation data including:
            - elevation_grid: 2D array of elevation values
            - bounds: Geographic boundaries
            - resolution: Spatial resolution of the data
            - metadata: Additional information about the dataset
            Returns empty dict if no data found.
            
        Raises:
            ValueError: If location_id is invalid
            RequestException: If API request fails
            
        Example:
            >>> client = CoREStackClient()
            >>> elevation = client.get_elevation_data('V001')
            >>> print(f"Elevation data shape: {elevation.get('elevation_grid', {}).get('shape', 'N/A')}")
        """
        # Input validation
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("location_id must be a non-empty string")
        
        location_id = location_id.strip()
        
        self.logger.info(f"Fetching elevation data for location '{location_id}'")
        
        try:
            params: Dict[str, str] = {"location_id": location_id}
            response = self._make_request("elevation", params)
            
            # Extract and validate data
            data = response.get("data", {})
            
            if not isinstance(data, dict):
                self.logger.warning(f"Expected dict from elevation endpoint, got {type(data)}")
                return {}
            
            # Validate elevation data structure
            if 'elevation_grid' in data:
                grid_data = data['elevation_grid']
                if not isinstance(grid_data, (dict, list)):
                    self.logger.warning(f"Invalid elevation_grid format: {type(grid_data)}")
                    data['elevation_grid'] = {}
            
            self.logger.info(f"Retrieved elevation data for location '{location_id}'")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch elevation data: {e}", exc_info=True)
            raise
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache usage.
        
        Returns:
            Dictionary containing cache statistics including:
            - cache_size: Number of cached entries
            - cache_ttl: Time-to-live in seconds
            - oldest_entry: Age of oldest cache entry
            - newest_entry: Age of newest cache entry
            - memory_usage: Estimated memory usage in bytes
        """
        if not self._cache_timestamps:
            return {
                "cache_size": 0,
                "cache_ttl": self.cache_ttl,
                "oldest_entry": None,
                "newest_entry": None,
                "memory_usage": 0
            }
        
        now = datetime.now()
        timestamps = list(self._cache_timestamps.values())
        
        oldest_age = (now - min(timestamps)).total_seconds()
        newest_age = (now - max(timestamps)).total_seconds()
        
        # Estimate memory usage (rough approximation)
        import sys
        cache_memory = sum(sys.getsizeof(v) for v in self._cache.values())
        timestamps_memory = sum(sys.getsizeof(v) for v in self._cache_timestamps.values())
        total_memory = cache_memory + timestamps_memory
        
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self.cache_ttl,
            "oldest_entry_age_seconds": oldest_age,
            "newest_entry_age_seconds": newest_age,
            "estimated_memory_usage_bytes": total_memory
        }


# Utility functions for backward compatibility
def get_spatial_units(unit_type: str = "village", state: Optional[str] = None) -> List[Dict]:
    """Get spatial units using default client."""
    client = CoREStackClient()
    return client.get_spatial_units(unit_type, state)


def get_seasonal_water_data(
    location_id: str, 
    start_year: int, 
    end_year: int,
    seasons: Optional[List[str]] = None
) -> Dict:
    """Get seasonal water data using default client."""
    client = CoREStackClient()
    return client.get_seasonal_water_data(location_id, start_year, end_year, seasons)
