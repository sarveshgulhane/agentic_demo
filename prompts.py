ROUTING_PROMPT = """You are an AI assistant whose task is to classify user queries.

The user query will fall into either "weather or "other":

WEATHER  
   - The user is asking about weather conditions, forecasts, temperature, rain, snow, climate, or atmospheric conditions.
   - The query may mention a city, country, region, or place.
   - Examples: 
     - "What's the weather in Paris?"
     - "Will it rain tomorrow in New York?"
     - "Temperature in Tokyo today"
   - Return:
     - "weather"

OTHER
   - Everything else

Rules:
- Choose ONLY one category out off weather or other.
- Do NOT explain your reasoning.
- Output must be a single category.

User query:
"{user_query}"

"""

CITY_PROMPT = """Identify the city mentioned in the sentence below. Respond only with the city name followed by its ISO 3166-1 two-letter country code.
Example:
What is the weather of Sydney?
Response: Sydney, AU

Sentence:
{user_query}
"""

WEATHER_PROMPT = """Analyze the weather data below and answer the weather question in plain language:
Example:
Weather in Pune:

- Conditions: Mostly cloudy with scattered clouds.
- Temperature: Warm, around 28.5°C (feels like about 27.2°C due 
to light wind).
- Wind: Light breeze (about 5 km/h), coming from the west-southwest 
direction.
- Visibility: Very clear (10 km).
- Humidity: Low at 23%.
- Pressure: Normal at 1009 hPa.
- Sunrise/Sunset: Sunrise is around 5:43 AM, sunset at 6:47 PM 
(local time in Pune).

The weather feels pleasant but could feel slightly cooler due to the wind 
chill.

Weather question:
{user_query}
Weather data:
{weather_data}
"""

DOCUMENT_ANSWER = """Analyze the data provided below and use it to answer the question. Refer only the given data.
Question:
{user_query}
Data context:
{rag_context}
"""
