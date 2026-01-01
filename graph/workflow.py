from langgraph.graph import StateGraph, START, END

from graph import AgentState
from graph.nodes import (
    decision_node_fn,
    city_node_fn,
    weather_node_fn,
    context_node_fn,
    answer_node_fn,
    evaluate_response,
)

flow_graph = StateGraph(state_schema=AgentState)

flow_graph.add_node("decision_node", decision_node_fn)

flow_graph.add_node("city_node", city_node_fn)
flow_graph.add_node("weather_node", weather_node_fn)

flow_graph.add_node("context_node", context_node_fn)

flow_graph.add_node("answer_node", answer_node_fn)

flow_graph.add_node("evaluation_node", evaluate_response)

flow_graph.add_conditional_edges(
    "decision_node",
    lambda state: "city_edge"
    if state.get("route", "").lower() == "weather"
    else "rag_edge",
    {"city_edge": "city_node", "rag_edge": "context_node"},
)

flow_graph.add_edge("city_node", "weather_node")
flow_graph.add_edge("weather_node", "answer_node")

flow_graph.add_edge("context_node", "answer_node")

flow_graph.add_edge("answer_node", "evaluation_node")

flow_graph.add_edge(START, "decision_node")
flow_graph.add_edge("evaluation_node", END)

app = flow_graph.compile()
