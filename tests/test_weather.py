"""
Test cases for weather API handling and external API integration.
"""
import pytest
from unittest.mock import patch, MagicMock
from services.weather_service import fetch_weather


class TestWeatherAPIHandling:
    """Test suite for weather API handling."""

    def test_fetch_weather_success(self, mock_weather_api_response):
        """Test successful weather API call."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "main": {
                    "temp": mock_weather_api_response["temperature"],
                    "humidity": mock_weather_api_response["humidity"],
                },
                "weather": [{"description": mock_weather_api_response["description"]}],
                "wind": {"speed": mock_weather_api_response["wind_speed"]},
                "name": mock_weather_api_response["location"],
            }
            mock_get.return_value = mock_response

            result = fetch_weather("New York")

            assert result is not None
            assert isinstance(result, dict)

    def test_fetch_weather_invalid_city(self):
        """Test weather API with invalid city name."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_get.return_value = mock_response

            with pytest.raises(Exception):
                fetch_weather("NonexistentCity12345")

    def test_fetch_weather_api_error(self):
        """Test weather API error handling."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_get.side_effect = Exception("API connection error")

            with pytest.raises(Exception):
                fetch_weather("New York")

    def test_fetch_weather_returns_dict(self, mock_weather_api_response):
        """Test that weather service returns a dictionary."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "main": {
                    "temp": 22,
                    "humidity": 65,
                },
                "weather": [{"description": "Clear"}],
                "wind": {"speed": 12},
                "name": "New York",
            }
            mock_get.return_value = mock_response

            result = fetch_weather("New York")

            assert isinstance(result, dict)

    def test_fetch_weather_contains_required_fields(self):
        """Test that weather response contains required fields."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "main": {"temp": 22, "humidity": 65},
                "weather": [{"description": "Clear"}],
                "wind": {"speed": 12},
                "name": "New York",
            }
            mock_get.return_value = mock_response

            result = fetch_weather("New York")

            assert "main" in result
            assert "weather" in result
            assert "wind" in result

    def test_fetch_weather_timeout(self):
        """Test weather API timeout handling."""
        with patch("services.weather_service.requests.get") as mock_get:
            import requests

            mock_get.side_effect = requests.Timeout("API timeout")

            with pytest.raises(requests.Timeout):
                fetch_weather("New York")

    def test_fetch_weather_different_cities(self):
        """Test weather API with different city names."""
        cities = ["London", "Tokyo", "Paris", "Sydney"]

        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "main": {"temp": 20, "humidity": 60},
                "weather": [{"description": "Cloudy"}],
                "wind": {"speed": 10},
                "name": "Test City",
            }
            mock_get.return_value = mock_response

            for city in cities:
                result = fetch_weather(city)
                assert result is not None

    def test_fetch_weather_empty_city_name(self):
        """Test weather API with empty city name."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_get.side_effect = ValueError("City name cannot be empty")

            with pytest.raises(ValueError):
                fetch_weather("")

    def test_fetch_weather_special_characters_in_name(self):
        """Test weather API with special characters in city name."""
        with patch("services.weather_service.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "main": {"temp": 20, "humidity": 60},
                "weather": [{"description": "Clear"}],
                "wind": {"speed": 10},
                "name": "São Paulo",
            }
            mock_get.return_value = mock_response

            result = fetch_weather("São Paulo")
            assert result is not None
