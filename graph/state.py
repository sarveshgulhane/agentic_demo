from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    # Core Input
    user_query: str

    # Routing: "weather" | "other"
    route: Optional[str]

    # Weather Tool
    weather_city: Optional[str]
    weather_data: Optional[Dict[str, Any]]

    # RAG Pipeline
    retrieved_chunks: Optional[List[str]]
    rag_context: Optional[str]

    # LLM Processing
    llm_input: Optional[str]
    llm_response: Optional[str]

    # Final Output
    final_answer: Optional[str]

    # Evaluation
    evaluation_metrics: Optional[Dict[str, Any]]

    # Debug / Observability
    trace_id: Optional[str]
    errors: Optional[List[str]]
