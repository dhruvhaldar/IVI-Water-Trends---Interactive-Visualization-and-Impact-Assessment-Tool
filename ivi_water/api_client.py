"""
CoRE Stack API Client Module

This module provides functions to interact with CoRE Stack REST APIs
for fetching spatial units and seasonal surface water data.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class CoREStackClient:
    """
    Client for interacting with CoRE Stack APIs.
    
    Handles authentication, caching, and rate limiting for API requests.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the CoRE Stack API client.
        
        Args:
            api_key: API key for authentication. If None, reads from environment.
            base_url: Base URL for the API. If None, uses default CoRE Stack URL.
        """
        self.api_key = api_key or os.getenv('CORE_API_KEY')
        self.base_url = base_url or os.getenv('CORE_API_BASE_URL', 'https://api.corestack.org/v1')
        
        if not self.api_key:
            raise ValueError("API key is required. Set CORE_API_KEY environment variable or pass api_key parameter.")
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'IVI-Water-Trends/0.1.0'
        })
        
        # Simple in-memory cache
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour default
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        return datetime.now() - cache_time < timedelta(seconds=self.cache_ttl)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if valid."""
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Dict) -> None:
        """Set data in cache."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Dict:
        """
        Make a request to the CoRE Stack API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_cache: Whether to use caching for this request
            
        Returns:
            JSON response data
            
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        cache_key = f"{url}_{json.dumps(params or {}, sort_keys=True)}"
        
        # Try cache first
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data
        
        try:
            logger.info(f"Making request to {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            if use_cache:
                self._set_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_spatial_units(self, unit_type: str = "village", state: Optional[str] = None) -> List[Dict]:
        """
        Get available spatial units from CoRE Stack.
        
        Args:
            unit_type: Type of spatial unit ('village', 'micro-watershed', 'tehsil')
            state: Optional state filter
            
        Returns:
            List of spatial unit information
        """
        params = {"type": unit_type}
        if state:
            params["state"] = state
        
        response = self._make_request("spatial-units", params)
        return response.get("data", [])
    
    def get_seasonal_water_data(
        self, 
        location_id: str, 
        start_year: int, 
        end_year: int,
        seasons: Optional[List[str]] = None
    ) -> Dict:
        """
        Get seasonal surface water data for a specific location.
        
        Args:
            location_id: Spatial unit identifier
            start_year: Start year for data
            end_year: End year for data
            seasons: List of seasons ('perennial', 'winter', 'monsoon'). If None, gets all.
            
        Returns:
            Dictionary containing seasonal water area time series data
        """
        params = {
            "location_id": location_id,
            "start_year": start_year,
            "end_year": end_year
        }
        
        if seasons:
            params["seasons"] = ",".join(seasons)
        
        response = self._make_request("water-trends/seasonal", params)
        return response.get("data", {})
    
    def get_water_trends_summary(
        self, 
        location_ids: List[str], 
        start_year: int, 
        end_year: int
    ) -> Dict:
        """
        Get summary statistics for multiple locations.
        
        Args:
            location_ids: List of spatial unit identifiers
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            Dictionary containing summary statistics
        """
        params = {
            "location_ids": ",".join(location_ids),
            "start_year": start_year,
            "end_year": end_year
        }
        
        response = self._make_request("water-trends/summary", params)
        return response.get("data", {})
    
    def get_elevation_data(self, location_id: str) -> Dict:
        """
        Get elevation data for a location (for 3D visualization).
        
        Args:
            location_id: Spatial unit identifier
            
        Returns:
            Dictionary containing elevation data
        """
        params = {"location_id": location_id}
        response = self._make_request("elevation", params)
        return response.get("data", {})
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cache usage."""
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self.cache_ttl,
            "cached_keys": list(self._cache.keys())
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
