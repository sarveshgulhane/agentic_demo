from graph import AgentState
from prompts import WEATHER_PROMPT, DOCUMENT_ANSWER
from services import get_llm_response


def answer_node_fn(state: AgentState) -> AgentState:
    """Generate answer from context."""
    try:
        if state["route"].lower() == "weather":
            state["llm_input"] = WEATHER_PROMPT.format(
                user_query=state["user_query"],
                weather_data=state["weather_data"],
            )
            state["llm_response"] = get_llm_response(state["llm_input"])

            return state
        state["llm_input"] = DOCUMENT_ANSWER.format(
            user_query=state["user_query"],
            rag_context=state["rag_context"],
        )
        state["llm_response"] = get_llm_response(state["llm_input"])

        return state
    except Exception as e:
        return {"errors": [str(e)]}
