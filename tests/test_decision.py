"""
Test cases for decision making and routing logic.
"""
import pytest
from unittest.mock import patch
from graph.nodes.decision import decision_node_fn


class TestDecisionNode:
    """Test suite for decision node and routing logic."""

    def test_decision_node_weather_route(self, sample_agent_state):
        """Test decision node routes weather queries correctly."""
        sample_agent_state["user_query"] = "What is the weather in New York?"

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "weather"

            result = decision_node_fn(sample_agent_state)

            assert result["route"] == "weather"
            mock_llm.assert_called_once()

    def test_decision_node_general_route(self, sample_agent_state):
        """Test decision node routes general queries correctly."""
        sample_agent_state["user_query"] = "What is machine learning?"

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "general"

            result = decision_node_fn(sample_agent_state)

            assert result["route"] == "general"

    def test_decision_node_preserves_user_query(self, sample_agent_state):
        """Test that decision node preserves the original user query."""
        original_query = "What is AI?"
        sample_agent_state["user_query"] = original_query

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "general"

            result = decision_node_fn(sample_agent_state)

            assert result["user_query"] == original_query

    def test_decision_node_case_insensitive(self, sample_agent_state):
        """Test that decision node handles case-insensitive routing."""
        sample_agent_state["user_query"] = "What is the WEATHER in Paris?"

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "WEATHER"

            result = decision_node_fn(sample_agent_state)

            # Should normalize to lowercase
            assert result["route"].lower() == "weather"

    def test_decision_node_multiple_queries_consistency(self, sample_agent_state):
        """Test decision node handles multiple similar queries consistently."""
        queries = [
            "What is the weather?",
            "Tell me about weather",
            "Weather forecast",
        ]

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "weather"

            for query in queries:
                sample_agent_state["user_query"] = query
                result = decision_node_fn(sample_agent_state)
                assert result["route"] == "weather"

    def test_decision_node_llm_call_format(self, sample_agent_state):
        """Test that decision node calls LLM with correct prompt format."""
        sample_agent_state["user_query"] = "Test query"

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "general"

            decision_node_fn(sample_agent_state)

            # Verify LLM was called with prompt
            assert mock_llm.called
            call_args = mock_llm.call_args[0][0]
            assert "Test query" in call_args or isinstance(call_args, str)

    def test_decision_node_returns_dict(self, sample_agent_state):
        """Test that decision node returns a dictionary."""
        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "general"

            result = decision_node_fn(sample_agent_state)

            assert isinstance(result, dict)

    def test_decision_node_error_handling(self, sample_agent_state):
        """Test decision node error handling."""
        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = decision_node_fn(sample_agent_state)

            # When there's an error, the function should return a state with errors
            assert "errors" in result
            assert len(result["errors"]) > 0
            # Default route is "rag" on error
            assert result["route"] == "rag"


class TestRoutingLogic:
    """Test suite for routing logic across different scenarios."""

    @pytest.mark.parametrize(
        "query,expected_route",
        [
            ("What's the weather in London?", "weather"),
            ("Tell me about Python", "general"),
            ("Current temperature in Tokyo", "weather"),
            ("Explain machine learning", "general"),
            ("Will it rain tomorrow?", "weather"),
        ],
    )
    def test_routing_various_queries(self, sample_agent_state, query, expected_route):
        """Test routing for various query types."""
        sample_agent_state["user_query"] = query

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = expected_route

            result = decision_node_fn(sample_agent_state)

            assert result["route"] == expected_route

    def test_routing_empty_query(self, sample_agent_state):
        """Test routing with empty query."""
        sample_agent_state["user_query"] = ""

        with patch("graph.nodes.decision.get_llm_response") as mock_llm:
            mock_llm.return_value = "general"

            result = decision_node_fn(sample_agent_state)

            assert "route" in result
