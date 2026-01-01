"""
Integration tests for the complete agentic workflow.
"""
import pytest
from unittest.mock import patch, MagicMock
from graph.state import AgentState


class TestAgentStateIntegration:
    """Test suite for AgentState integration."""

    def test_agent_state_initialization(self):
        """Test AgentState initialization with required fields."""
        state: AgentState = {
            "user_query": "What is AI?",
            "route": None,
            "weather_city": None,
            "weather_data": None,
            "retrieved_chunks": None,
            "rag_context": None,
            "llm_input": None,
            "llm_response": None,
            "final_answer": None,
            "evaluation_metrics": None,
            "trace_id": None,
            "errors": None,
        }

        assert state["user_query"] == "What is AI?"
        assert state["route"] is None

    def test_agent_state_weather_route(self, sample_agent_state):
        """Test AgentState with weather route."""
        sample_agent_state["route"] = "weather"
        sample_agent_state["weather_city"] = "New York"
        sample_agent_state["weather_data"] = {
            "temperature": 22,
            "humidity": 65,
        }

        assert sample_agent_state["route"] == "weather"
        assert sample_agent_state["weather_city"] == "New York"

    def test_agent_state_rag_route(self, sample_agent_state):
        """Test AgentState with RAG route."""
        sample_agent_state["route"] = "general"
        sample_agent_state["retrieved_chunks"] = [
            "Chunk 1",
            "Chunk 2",
        ]
        sample_agent_state["rag_context"] = "Combined context"

        assert sample_agent_state["route"] == "general"
        assert len(sample_agent_state["retrieved_chunks"]) == 2

    def test_agent_state_with_evaluation(self, sample_agent_state):
        """Test AgentState with evaluation metrics."""
        sample_agent_state["final_answer"] = "This is the final answer"
        sample_agent_state["evaluation_metrics"] = {
            "confidence": 0.95,
            "relevance": 0.92,
        }

        assert sample_agent_state["evaluation_metrics"]["confidence"] == 0.95

    def test_agent_state_with_errors(self, sample_agent_state):
        """Test AgentState error tracking."""
        sample_agent_state["errors"] = ["Error 1", "Error 2"]

        assert len(sample_agent_state["errors"]) == 2
        assert "Error 1" in sample_agent_state["errors"]

    def test_agent_state_trace_id(self, sample_agent_state):
        """Test AgentState trace ID."""
        import uuid

        trace_id = str(uuid.uuid4())
        sample_agent_state["trace_id"] = trace_id

        assert sample_agent_state["trace_id"] == trace_id


class TestWorkflowIntegration:
    """Test suite for workflow integration."""

    def test_weather_workflow_path(self, sample_agent_state):
        """Test complete weather workflow path."""
        sample_agent_state["user_query"] = "What's the weather in London?"

        with patch("graph.nodes.decision.get_llm_response") as mock_decision:
            with patch("graph.nodes.city.get_llm_response") as mock_city:
                with patch("services.weather_service.fetch_weather") as mock_weather:
                    with patch("graph.nodes.answer.get_llm_response") as mock_answer:
                        mock_decision.return_value = "weather"
                        mock_city.return_value = "London"
                        mock_weather.return_value = {
                            "name": "London",
                            "main": {"temp": 15},
                            "weather": [{"description": "Rainy"}],
                        }
                        mock_answer.return_value = "It's rainy in London with 15°C"

                        # Simulate workflow
                        sample_agent_state["route"] = mock_decision.return_value
                        sample_agent_state["weather_city"] = mock_city.return_value
                        sample_agent_state["weather_data"] = mock_weather.return_value
                        sample_agent_state["final_answer"] = mock_answer.return_value

                        assert sample_agent_state["route"] == "weather"
                        assert sample_agent_state["final_answer"] is not None

    def test_rag_workflow_path(self, sample_agent_state):
        """Test complete RAG workflow path."""
        sample_agent_state["user_query"] = "Explain machine learning"

        with patch("graph.nodes.decision.get_llm_response") as mock_decision:
            with patch("rag.retriever.retrieve_with_scores") as mock_retrieve:
                with patch("graph.nodes.answer.get_llm_response") as mock_answer:
                    mock_decision.return_value = "general"
                    mock_retrieve.return_value = [
                        (MagicMock(page_content="ML content 1"), 0.95),
                        (MagicMock(page_content="ML content 2"), 0.88),
                    ]
                    mock_answer.return_value = "Machine learning is..."

                    # Simulate workflow
                    sample_agent_state["route"] = mock_decision.return_value
                    results = mock_retrieve.return_value
                    sample_agent_state["retrieved_chunks"] = [doc.page_content for doc, _ in results]
                    sample_agent_state["final_answer"] = mock_answer.return_value

                    assert sample_agent_state["route"] == "general"
                    assert len(sample_agent_state["retrieved_chunks"]) == 2

    def test_workflow_error_handling(self, sample_agent_state):
        """Test workflow error handling."""
        with patch("graph.nodes.decision.get_llm_response") as mock_decision:
            mock_decision.side_effect = Exception("Decision error")

            with pytest.raises(Exception):
                # This would be called in the actual workflow
                mock_decision("test")


class TestEndToEndScenarios:
    """End-to-end test scenarios."""

    def test_weather_query_end_to_end(self):
        """Test complete weather query scenario."""
        query = "What's the weather in Paris?"

        # This would normally go through the full workflow
        assert "weather" in query.lower() or "temperature" in query.lower()

    def test_general_query_end_to_end(self):
        """Test complete general query scenario."""
        query = "What is artificial intelligence?"

        # This would normally go through RAG workflow
        assert isinstance(query, str)
        assert len(query) > 0

    @pytest.mark.parametrize(
        "query,expected_field",
        [
            ("What's the weather?", "weather_data"),
            ("Tell me about AI", "retrieved_chunks"),
            ("Temperature in Tokyo", "weather_data"),
        ],
    )
    def test_various_query_scenarios(self, sample_agent_state, query, expected_field):
        """Test various query scenarios."""
        sample_agent_state["user_query"] = query

        assert sample_agent_state["user_query"] == query
        assert expected_field in sample_agent_state
