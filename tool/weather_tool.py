from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests

class WeatherToolInput(BaseModel):
    location: str = Field(..., description="City name, optionally with state and country (e.g., 'Kanchipuram, Tamil Nadu, IN')")

class WeatherTool(BaseTool):
    name: str = "weather_tool"
    description: str = "Fetches current weather data from OpenWeatherMap API using city name and optional state/country."
    args_schema = WeatherToolInput

    def _run(self, location: str) -> str:
        api_key = "f2c5366300d4faa33357fae18a0c4ca4"
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"  # Use "imperial" for Fahrenheit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            weather_description = data['weather'][0]['description'].capitalize()
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            city = data['name']
            country = data['sys']['country']

            return (f"Weather info for {city}, {country}:\n"
                    f"Condition: {weather_description}\n"
                    f"Temperature: {temperature}°C\n"
                    f"Humidity: {humidity}%\n"
                    f"Wind Speed: {wind_speed} m/s")
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"
        except Exception as err:
            return f"An error occurred: {err}"
