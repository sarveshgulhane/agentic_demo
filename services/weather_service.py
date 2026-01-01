import requests

from config import OPENWEATHER_API_KEY

url = "https://api.openweathermap.org/data/2.5/weather"


def fetch_weather(city: str) -> dict:
    """OpenWeatherMap API service."""

    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()
