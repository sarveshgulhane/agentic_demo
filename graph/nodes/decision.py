from graph import AgentState
from prompts import ROUTING_PROMPT
from services import get_llm_response


def decision_node_fn(state: AgentState) -> AgentState:
    """Route between Weather and RAG based on query."""
    try:
        route = get_llm_response(
            ROUTING_PROMPT.format(user_query=state["user_query"])
        ).lower().strip()
        state["route"] = route
        return state
    except Exception as e:
        state["errors"] = [str(e)]
        state["route"] = "rag"
        return state
