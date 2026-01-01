from graph import AgentState
from prompts import CITY_PROMPT
from services import get_llm_response


def city_node_fn(state: AgentState) -> AgentState:
    """Extract City name with Country code from query."""
    try:
        state["weather_city"] = get_llm_response(
            CITY_PROMPT.format(user_query=state["user_query"])
        )
        return state

    except Exception as e:
        return {"errors": [str(e)]}
