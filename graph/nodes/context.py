from graph import AgentState
from rag import retrieve_with_scores


def context_node_fn(state: AgentState) -> AgentState:
    """RAG orchestration, Retrieve and process documents."""
    try:
        docs = retrieve_with_scores(query=state["user_query"], k=3)

        state["retrieved_chunks"] = [doc[0].page_content for doc in docs]
        state["rag_context"] = "\n\n".join(state["retrieved_chunks"])

        return state
    except Exception as e:
        return {"errors": [str(e)]}
