"""
Test cases for LLM service and processing.
"""
import pytest
from unittest.mock import patch
from services.llm_service import get_llm_response


class TestLLMService:
    """Test suite for LLM service."""

    def test_get_llm_response_success(self, mock_chat_ollama):
        """Test successful LLM response."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = "What is machine learning?"
            response = get_llm_response(prompt)

            assert response == "This is a mock LLM response."
            mock_chat_ollama.invoke.assert_called_once()

    def test_get_llm_response_with_complex_prompt(self, mock_chat_ollama):
        """Test LLM response with complex prompt."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = "Explain the concept of neural networks in detail, including backpropagation."
            response = get_llm_response(prompt)

            assert isinstance(response, str)
            assert len(response) > 0
            mock_chat_ollama.invoke.assert_called_once()

    def test_get_llm_response_empty_prompt(self, mock_chat_ollama):
        """Test LLM response with empty prompt."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = ""
            response = get_llm_response(prompt)

            assert response == "This is a mock LLM response."

    def test_get_llm_response_special_characters(self, mock_chat_ollama):
        """Test LLM response with special characters in prompt."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = "What is @#$%^&*()? Question?"
            response = get_llm_response(prompt)

            assert isinstance(response, str)
            mock_chat_ollama.invoke.assert_called_once()

    def test_get_llm_response_returns_string_type(self, mock_chat_ollama):
        """Test that LLM response is always a string."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = "Describe AI"
            response = get_llm_response(prompt)

            assert isinstance(response, str)

    def test_get_llm_response_long_prompt(self, mock_chat_ollama):
        """Test LLM response with very long prompt."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            long_prompt = "What is AI? " * 100
            response = get_llm_response(long_prompt)

            assert isinstance(response, str)

    def test_get_llm_response_handles_error_gracefully(self, mock_chat_ollama):
        """Test LLM service error handling."""
        mock_chat_ollama.invoke.side_effect = Exception("LLM service error")

        with patch("services.llm_service.llm", mock_chat_ollama):
            with pytest.raises(Exception):
                get_llm_response("What is AI?")

    def test_get_llm_response_with_unicode(self, mock_chat_ollama):
        """Test LLM response with unicode characters."""
        with patch("services.llm_service.llm", mock_chat_ollama):
            prompt = "What is 你好世界 and مرحبا العالم?"
            response = get_llm_response(prompt)

            assert isinstance(response, str)
