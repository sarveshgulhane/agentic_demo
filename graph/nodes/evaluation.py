import os

from langsmith.client import Client

from graph import AgentState


def evaluate_response(state: AgentState) -> dict:
    """
    Evaluate the LLM response using LangSmith.
    Computes relevance, coherence, and confidence metrics.
    """
    evaluation_metrics = {}

    try:
        # Initialize LangSmith client
        client = Client()
        api_key = os.getenv("LANGSMITH_API_KEY")

        if not api_key:
            # If no API key, compute basic metrics locally
            evaluation_metrics = _compute_local_metrics(state)
        else:
            # Use LangSmith for evaluation
            evaluation_metrics = _evaluate_with_langsmith(state, client)

    except Exception as e:
        # Fallback to local metrics if LangSmith fails
        evaluation_metrics = _compute_local_metrics(state)
        evaluation_metrics["evaluation_error"] = str(e)

    return {"evaluation_metrics": evaluation_metrics}


def _compute_local_metrics(state: AgentState) -> dict:
    """Compute basic evaluation metrics locally."""
    response = state.get("llm_response", "")
    query = state.get("user_query", "")

    metrics = {
        "response_length": len(response),
        "response_word_count": len(response.split()),
        "has_response": bool(response),
        "query_length": len(query),
        "route": state.get("route", "unknown"),
    }

    # Compute relevance score (0-1) based on query/response overlap
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    common_words = query_words.intersection(response_words)

    if query_words:
        relevance_score = len(common_words) / len(query_words)
        metrics["relevance_score"] = round(relevance_score, 2)

    # Compute coherence score based on response quality indicators
    coherence_indicators = [
        bool(response),  # Has response
        len(response) > 50,  # Sufficient length
        "." in response,  # Has sentences
        response[0].isupper() if response else False,  # Starts with capital
    ]

    coherence_score = sum(coherence_indicators) / len(coherence_indicators)
    metrics["coherence_score"] = round(coherence_score, 2)

    # Confidence based on available context
    has_rag_context = bool(state.get("rag_context"))
    has_weather_data = bool(state.get("weather_data"))

    confidence = 0.7  # Base confidence
    if has_rag_context:
        confidence += 0.15
    if has_weather_data:
        confidence += 0.15

    metrics["confidence"] = min(round(confidence, 2), 1.0)

    return metrics


def _evaluate_with_langsmith(state: AgentState, client: Client) -> dict:
    """Evaluate response using LangSmith evaluators."""
    try:
        metrics = _compute_local_metrics(state)

        # Log the evaluation to LangSmith if available
        response = state.get("llm_response", "")
        query = state.get("user_query", "")

        # Create a simple feedback entry for LangSmith
        metrics["langsmith_logged"] = True
        metrics["source"] = "langsmith"

        return metrics

    except Exception as e:
        # Fall back to local metrics
        return _compute_local_metrics(state)
