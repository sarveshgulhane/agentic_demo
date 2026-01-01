from services import fetch_weather

from graph import AgentState


def weather_node_fn(state: AgentState) -> AgentState:
    """Handle weather related queries via API."""
    try:
        state["weather_data"] = fetch_weather(state["weather_city"])

        return state

    except Exception as e:
        return {"errors": [str(e)]}
