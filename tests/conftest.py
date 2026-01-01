"""
Pytest configuration and shared fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_llm_response():
    """Mock LLM response fixture."""
    return "This is a mock LLM response."


@pytest.fixture
def mock_embeddings():
    """Mock embeddings fixture."""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1] * 384  # 384-dim vector
    return embeddings


@pytest.fixture
def sample_agent_state():
    """Sample AgentState for testing."""
    return {
        "user_query": "What is the weather in New York?",
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


@pytest.fixture
def sample_retrieved_documents():
    """Sample retrieved documents from RAG."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "doc1.pdf", "page": 1},
            "score": 0.95,
        },
        {
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "doc2.pdf", "page": 2},
            "score": 0.92,
        },
        {
            "content": "Natural language processing helps computers understand text.",
            "metadata": {"source": "doc3.pdf", "page": 1},
            "score": 0.88,
        },
    ]


@pytest.fixture
def mock_weather_api_response():
    """Mock weather API response."""
    return {
        "location": "New York",
        "temperature": 22,
        "description": "Clear sky",
        "humidity": 65,
        "wind_speed": 12,
    }


@pytest.fixture
def mock_vector_store():
    """Mock QdrantVectorStore."""
    vector_store = MagicMock()
    vector_store.similarity_search_with_relevance_scores.return_value = [
        (MagicMock(page_content="Sample content 1"), 0.95),
        (MagicMock(page_content="Sample content 2"), 0.88),
    ]
    return vector_store


@pytest.fixture
def mock_chat_ollama():
    """Mock ChatOllama LLM."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a mock LLM response."
    llm.invoke.return_value = mock_response
    return llm
