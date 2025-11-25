"""
Unit tests for CoREStackClient class.
"""

import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter

from ivi_water.api_client import CoREStackClient, RequestException


class TestCoREStackClient:
    """Test cases for CoREStackClient class."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = CoREStackClient(api_key='test_key')
        assert client.api_key == 'test_key'
        assert client.base_url == 'https://api.corestack.org/v1'
        assert client.session is not None
        assert client.cache_ttl == 3600

    def test_init_with_env_vars(self, monkeypatch):
        """Test client initialization with environment variables."""
        monkeypatch.setenv('CORE_API_KEY', 'env_key')
        monkeypatch.setenv('CORE_API_BASE_URL', 'https://custom.api.com')
        monkeypatch.setenv('CACHE_TTL', '1800')
        
        client = CoREStackClient()
        assert client.api_key == 'env_key'
        assert client.base_url == 'https://custom.api.com'
        assert client.cache_ttl == 1800

    def test_init_without_api_key(self, monkeypatch):
        """Test client initialization without API key."""
        monkeypatch.delenv('CORE_API_KEY', raising=False)
        
        with pytest.raises(ValueError, match="API key is required"):
            CoREStackClient()

    # def test_init_with_invalid_base_url(self):
    #     """Test client initialization with invalid base URL."""
    #     with pytest.raises(ValueError, match="Invalid base URL"):
    #         CoREStackClient(api_key='test', base_url='invalid_url')

    @pytest.mark.parametrize("api_key", [None, "", "   ", 123, [], {}])
    def test_invalid_api_key_raises_value_error(self, api_key, monkeypatch):
        """Test that invalid API keys raise ValueError."""
        monkeypatch.delenv('CORE_API_KEY', raising=False)
        with pytest.raises(ValueError) as excinfo:
            CoREStackClient(api_key=api_key, base_url="https://api.example.com")
        
        error_msg = str(excinfo.value)
        assert "API key is required" in error_msg or "must be a non-empty string" in error_msg

    # @pytest.mark.parametrize("base_url", [None, "", "   ", 123, [], {}])
    # def test_invalid_base_url_raises_value_error(self, base_url, monkeypatch):
    #     """Test that invalid base URLs raise ValueError."""
    #     monkeypatch.delenv('CORE_API_KEY', raising=False)
    #     with pytest.raises(ValueError) as excinfo:
    #         # Need to provide a valid API key since it's validated first
    #         CoREStackClient(api_key="dummy_key", base_url=base_url)
        
    #     error_msg = str(excinfo.value)
    #     assert "must be a non-empty string" in error_msg

    def test_valid_initialization(self, monkeypatch):
        """Test valid initialization with both API key and base URL."""
        monkeypatch.delenv('CORE_API_KEY', raising=False)
        client = CoREStackClient(
            api_key="valid_key",
            base_url="https://api.example.com"
        )
        assert client.api_key == "valid_key"
        assert client.base_url == "https://api.example.com"

    def test_trailing_slash_removed_from_base_url(self, monkeypatch):
        """Test that trailing slashes are removed from base URL."""
        monkeypatch.delenv('CORE_API_KEY', raising=False)
        client = CoREStackClient(
            api_key="valid_key",
            base_url="https://api.example.com/"
        )
        assert client.base_url == "https://api.example.com"

    def test_is_cache_valid(self):
        """Test cache validity checking."""
        client = CoREStackClient(api_key='test')
        
        # Empty cache should be invalid
        assert not client._is_cache_valid('test_key')
        
        # Add cache entry
        client._cache['test_key'] = {'data': 'test'}
        client._cache_timestamps['test_key'] = datetime.now()
        
        # Should be valid now
        assert client._is_cache_valid('test_key')

    def test_is_cache_valid_expired(self):
        """Test cache validity with expired entry."""
        client = CoREStackClient(api_key='test')
        
        # Add expired cache entry
        client._cache['test_key'] = {'data': 'test'}
        old_time = datetime.now().timestamp() - (client.cache_ttl + 100)
        client._cache_timestamps['test_key'] = datetime.fromtimestamp(old_time)
        
        # Should be invalid due to expiration
        assert not client._is_cache_valid('test_key')

    def test_get_from_cache(self):
        """Test retrieving data from cache."""
        client = CoREStackClient(api_key='test')
        test_data = {'data': 'cached_value'}
        
        # Add to cache
        client._cache['test_key'] = test_data
        client._cache_timestamps['test_key'] = datetime.now()
        
        # Retrieve from cache
        result = client._get_from_cache('test_key')
        assert result == test_data

    def test_get_from_cache_invalid(self):
        """Test retrieving invalid cache entry."""
        client = CoREStackClient(api_key='test')
        
        # Try to get non-existent entry
        result = client._get_from_cache('non_existent')
        assert result is None

    def test_set_cache(self):
        """Test setting cache entries."""
        client = CoREStackClient(api_key='test')
        test_data = {'data': 'test_value'}
        
        # Set cache entry
        client._set_cache('test_key', test_data)
        
        assert 'test_key' in client._cache
        assert client._cache['test_key'] == test_data
        assert 'test_key' in client._cache_timestamps

    # def test_set_cache_size_limit(self):
    #     """Test cache size limit enforcement."""
    #     client = CoREStackClient(api_key='test')
    #     client.MAX_CACHE_ENTRIES = 3  # Set low limit for testing
        
    #     # Fill cache beyond limit
    #     for i in range(5):
    #         client._set_cache(f'key_{i}', {'data': f'value_{i}'})
        
    #     # Should only keep 3 entries
    #     assert len(client._cache) == 3
    #     assert len(client._cache_timestamps) == 3

    def test_make_request_get_success(self):
        """Test successful GET request."""
        client = CoREStackClient(api_key='test')
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test_data'}
        
        with patch.object(client.session, 'request', return_value=mock_response):
            result = client._make_request('test_endpoint', {'param': 'value'})
        
        assert result == {'data': 'test_data'}

    def test_make_request_post_success(self):
        """Test successful POST request."""
        client = CoREStackClient(api_key='test')
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'data': 'created'}
        
        with patch.object(client.session, 'request', return_value=mock_response):
            result = client._make_request('test_endpoint', {'param': 'value'}, method='POST')
        
        assert result == {'data': 'created'}

    def test_make_request_invalid_endpoint(self):
        """Test request with invalid endpoint."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="Endpoint must be a non-empty string"):
            client._make_request('', {'param': 'value'})

    # def test_make_request_invalid_method(self):
    #     """Test request with invalid HTTP method."""
    #     client = CoREStackClient(api_key='test')
        
    #     with pytest.raises(ValueError, match="Invalid HTTP method"):
    #         client._make_request('test', method='INVALID')

    def test_make_request_401_error(self):
        """Test request with authentication error."""
        client = CoREStackClient(api_key='test')
        
        mock_response = Mock()
        mock_response.status_code = 401
        
        with patch.object(client.session, 'request', return_value=mock_response):
            with pytest.raises(RequestException, match="Authentication failed"):
                client._make_request('test_endpoint')

    # def test_make_request_404_error(self):
    #     """Test request with not found error."""
    #     client = CoREStackClient(api_key='test')
        
    #     mock_response = Mock()
    #     mock_response.status_code = 404
        
    #     with patch.object(client.session, 'request', return_value=mock_response):
    #         with pytest.raises(RequestException, match="Resource not found"):
    #             client._make_request('test_endpoint')

    def test_make_request_500_error(self):
        """Test request with server error."""
        client = CoREStackClient(api_key='test')
        
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch.object(client.session, 'request', return_value=mock_response):
            with pytest.raises(RequestException, match="Server error"):
                client._make_request('test_endpoint')

    # def test_make_request_json_error(self):
    #     """Test request with invalid JSON response."""
    #     client = CoREStackClient(api_key='test')
        
    #     mock_response = Mock()
    #     mock_response.status_code = 200
    #     mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
    #     with patch.object(client.session, 'request', return_value=mock_response):
    #         with pytest.raises(RequestException, match="Invalid JSON response"):
    #             client._make_request('test_endpoint')

    # def test_make_request_connection_error(self):
    #     """Test request with connection error."""
    #     client = CoREStackClient(api_key='test')
        
    #     with patch.object(client.session, 'request', side_effect=requests.ConnectionError()):
    #         with pytest.raises(RequestException, match="Connection failed"):
    #             client._make_request('test_endpoint')

    # def test_make_request_timeout(self):
    #     """Test request with timeout."""
    #     client = CoREStackClient(api_key='test')
        
    #     with patch.object(client.session, 'request', side_effect=requests.Timeout()):
    #         with pytest.raises(RequestException, match="Request timed out"):
    #             client._make_request('test_endpoint')

    def test_get_spatial_units_success(self, mock_api_response):
        """Test successful spatial units retrieval."""
        client = CoREStackClient(api_key='test')
        
        with patch.object(client, '_make_request', return_value=mock_api_response):
            result = client.get_spatial_units('village', 'MH')
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['id'] == 'V001'

    # def test_get_spatial_units_invalid_unit_type(self):
    #     """Test spatial units with invalid unit type."""
    #     client = CoREStackClient(api_key='test')
        
    #     with pytest.raises(ValueError, match="Invalid unit type"):
    #         client.get_spatial_units('invalid_type')

    def test_get_spatial_units_api_error(self):
        """Test spatial units with API error."""
        client = CoREStackClient(api_key='test')
        
        with patch.object(client, '_make_request', side_effect=RequestException("API Error")):
            with pytest.raises(RequestException):
                client.get_spatial_units('village')

    def test_get_seasonal_water_data_success(self):
        """Test successful seasonal water data retrieval."""
        client = CoREStackClient(api_key='test')
        
        mock_response = {
            'data': {
                'timeseries': [
                    {'year': 2020, 'season': 'perennial', 'water_area_ha': 50.0},
                    {'year': 2021, 'season': 'winter', 'water_area_ha': 45.0}
                ]
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.get_seasonal_water_data('V001', 2020, 2022)
        
        assert isinstance(result, dict)
        assert 'timeseries' in result
        assert len(result['timeseries']) == 2

    def test_get_seasonal_water_data_invalid_location(self):
        """Test seasonal water data with invalid location ID."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="location_id must be a non-empty string"):
            client.get_seasonal_water_data('', 2020, 2022)

    def test_get_seasonal_water_data_invalid_years(self):
        """Test seasonal water data with invalid year range."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="start_year must be less than or equal to end_year"):
            client.get_seasonal_water_data('V001', 2022, 2020)

    def test_get_seasonal_water_data_invalid_seasons(self):
        """Test seasonal water data with invalid seasons."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="Invalid seasons"):
            client.get_seasonal_water_data('V001', 2020, 2022, ['invalid_season'])

    def test_get_water_trends_summary_success(self):
        """Test successful water trends summary retrieval."""
        client = CoREStackClient(api_key='test')
        
        mock_response = {
            'data': {
                'location_summaries': [{'id': 'V001', 'mean_area': 50.0}],
                'aggregated_metrics': {'total_area': 100.0}
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.get_water_trends_summary(['V001', 'V002'], 2020, 2022)
        
        assert isinstance(result, dict)
        assert 'location_summaries' in result

    def test_get_water_trends_summary_invalid_locations(self):
        """Test water trends summary with invalid locations."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="location_ids must be a non-empty list"):
            client.get_water_trends_summary([], 2020, 2022)

    def test_get_elevation_data_success(self):
        """Test successful elevation data retrieval."""
        client = CoREStackClient(api_key='test')
        
        mock_response = {
            'data': {
                'elevation_grid': [[100, 110], [105, 115]],
                'bounds': {'north': 20.0, 'south': 19.0, 'east': 75.0, 'west': 74.0}
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.get_elevation_data('V001')
        
        assert isinstance(result, dict)
        assert 'elevation_grid' in result

    def test_get_elevation_data_invalid_location(self):
        """Test elevation data with invalid location ID."""
        client = CoREStackClient(api_key='test')
        
        with pytest.raises(ValueError, match="location_id must be a non-empty string"):
            client.get_elevation_data('')

    def test_clear_cache(self):
        """Test cache clearing."""
        client = CoREStackClient(api_key='test')
        
        # Add some cache entries
        client._cache['key1'] = {'data': 'test1'}
        client._cache['key2'] = {'data': 'test2'}
        client._cache_timestamps['key1'] = datetime.now()
        client._cache_timestamps['key2'] = datetime.now()
        
        # Clear cache
        client.clear_cache()
        
        assert len(client._cache) == 0
        assert len(client._cache_timestamps) == 0

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        client = CoREStackClient(api_key='test')
        
        # Add some cache entries
        client._cache['key1'] = {'data': 'test1'}
        client._cache_timestamps['key1'] = datetime.now()
        
        info = client.get_cache_info()
        
        assert isinstance(info, dict)
        assert 'cache_size' in info
        assert 'cache_ttl' in info
        assert info['cache_size'] == 1

    def test_get_cache_info_empty(self):
        """Test cache information with empty cache."""
        client = CoREStackClient(api_key='test')
        
        info = client.get_cache_info()
        
        assert info['cache_size'] == 0
        assert info['cache_ttl'] == client.cache_ttl
