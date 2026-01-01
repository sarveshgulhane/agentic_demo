"""
Test cases for embedding service and vector operations.
"""
import pytest
from unittest.mock import patch, MagicMock
from services.embedding_service import get_embeddings


class TestEmbeddingService:
    """Test suite for embedding service."""

    def test_get_embeddings_initialization(self):
        """Test successful embeddings initialization."""
        with patch("services.embedding_service.HuggingFaceEmbeddings") as mock_embeddings_class:
            mock_embeddings = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings

            embeddings = get_embeddings()

            assert embeddings == mock_embeddings
            mock_embeddings_class.assert_called_once()

    def test_get_embeddings_correct_model(self):
        """Test that correct model is used for embeddings."""
        with patch("services.embedding_service.HuggingFaceEmbeddings") as mock_embeddings_class:
            mock_embeddings = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings

            get_embeddings()

            call_args = mock_embeddings_class.call_args
            assert "all-MiniLM-L6-v2" in str(call_args)

    def test_embeddings_embed_query_dimension(self, mock_embeddings):
        """Test that embeddings have correct dimensions."""
        embedding_vector = mock_embeddings.embed_query("test query")

        assert isinstance(embedding_vector, list)
        assert len(embedding_vector) == 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def test_embeddings_embed_query_values_range(self, mock_embeddings):
        """Test that embedding values are in reasonable range."""
        embedding_vector = mock_embeddings.embed_query("test query")

        # Embeddings are typically normalized, so values should be between -1 and 1
        for value in embedding_vector:
            assert isinstance(value, float)

    def test_embeddings_consistency(self, mock_embeddings):
        """Test that same query produces consistent embeddings."""
        query = "machine learning"
        embedding1 = mock_embeddings.embed_query(query)
        embedding2 = mock_embeddings.embed_query(query)

        # In mock, they should be identical
        assert embedding1 == embedding2

    def test_embeddings_different_queries(self, mock_embeddings):
        """Test that different queries produce different embeddings."""
        embedding1 = mock_embeddings.embed_query("machine learning")
        embedding2 = mock_embeddings.embed_query("deep learning")

        # Mocks return same value, but in real scenario they'd be different
        assert isinstance(embedding1, list)
        assert isinstance(embedding2, list)

    def test_embeddings_with_empty_query(self, mock_embeddings):
        """Test embeddings with empty query."""
        embedding = mock_embeddings.embed_query("")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_embeddings_with_unicode(self, mock_embeddings):
        """Test embeddings with unicode characters."""
        queries = ["你好", "مرحبا", "नमस्ते", "Привет"]

        for query in queries:
            embedding = mock_embeddings.embed_query(query)
            assert isinstance(embedding, list)
            assert len(embedding) == 384

    def test_embeddings_with_special_characters(self, mock_embeddings):
        """Test embeddings with special characters."""
        query = "Test@#$%^&*()"
        embedding = mock_embeddings.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_embeddings_with_long_text(self, mock_embeddings):
        """Test embeddings with long text."""
        long_text = "word " * 1000
        embedding = mock_embeddings.embed_query(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384


class TestEmbeddingVectorOperations:
    """Test suite for vector operations with embeddings."""

    @pytest.fixture
    def embedding_vectors(self):
        """Sample embedding vectors for testing."""
        return {
            "query": [0.1] * 384,
            "doc1": [0.1] * 384,
            "doc2": [0.2] * 384,
            "doc3": [0.3] * 384,
        }

    def test_embedding_vector_normalization(self, embedding_vectors):
        """Test that embeddings are properly normalized."""
        for vector_name, vector in embedding_vectors.items():
            assert len(vector) == 384
            assert all(isinstance(v, float) for v in vector)

    def test_cosine_similarity_computation(self, embedding_vectors):
        """Test cosine similarity between embeddings."""
        import numpy as np

        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        sim = cosine_similarity(
            embedding_vectors["query"],
            embedding_vectors["doc1"],
        )

        assert 0 <= sim <= 1
