"""
Test cases for RAG (Retrieval-Augmented Generation) and retrieval logic.
"""
import pytest
from unittest.mock import patch, MagicMock
from rag.retriever import get_vector_store, retrieve_with_scores


class TestRetrieval:
    """Test suite for retrieval logic."""

    def test_get_vector_store_success(self, mock_embeddings):
        """Test successful vector store initialization."""
        with patch("rag.retriever.get_embeddings", return_value=mock_embeddings):
            with patch("rag.retriever.QdrantVectorStore.from_existing_collection") as mock_from_collection:
                mock_vector_store = MagicMock()
                mock_from_collection.return_value = mock_vector_store

                vector_store = get_vector_store(collection_name="documents")

                assert vector_store == mock_vector_store
                mock_from_collection.assert_called_once()

    def test_get_vector_store_with_custom_collection(self, mock_embeddings):
        """Test vector store with custom collection name."""
        with patch("rag.retriever.get_embeddings", return_value=mock_embeddings):
            with patch("rag.retriever.QdrantVectorStore.from_existing_collection") as mock_from_collection:
                mock_vector_store = MagicMock()
                mock_from_collection.return_value = mock_vector_store

                collection_name = "custom_collection"
                get_vector_store(collection_name=collection_name)

                call_args = mock_from_collection.call_args
                assert call_args.kwargs["collection_name"] == collection_name

    def test_retrieve_with_scores_success(self):
        """Test successful document retrieval with scores."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_doc1 = MagicMock()
            mock_doc1.page_content = "Content 1"
            mock_doc2 = MagicMock()
            mock_doc2.page_content = "Content 2"

            mock_vector_store.similarity_search_with_relevance_scores.return_value = [
                (mock_doc1, 0.95),
                (mock_doc2, 0.87),
            ]
            mock_get_store.return_value = mock_vector_store

            query = "machine learning basics"
            results = retrieve_with_scores(query, k=2)

            assert len(results) == 2
            assert results[0][1] == 0.95
            assert results[1][1] == 0.87

    def test_retrieve_with_scores_default_k(self):
        """Test retrieve with default k=3."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = []
            mock_get_store.return_value = mock_vector_store

            retrieve_with_scores("test query")

            call_args = mock_vector_store.similarity_search_with_relevance_scores.call_args
            assert call_args.kwargs["k"] == 3

    def test_retrieve_with_scores_custom_k(self):
        """Test retrieve with custom k value."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = []
            mock_get_store.return_value = mock_vector_store

            retrieve_with_scores("test query", k=5)

            call_args = mock_vector_store.similarity_search_with_relevance_scores.call_args
            assert call_args.kwargs["k"] == 5

    def test_retrieve_with_scores_returns_list(self):
        """Test that retrieve returns a list."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = [
                (MagicMock(), 0.9),
            ]
            mock_get_store.return_value = mock_vector_store

            results = retrieve_with_scores("query")

            assert isinstance(results, list)

    def test_retrieve_with_scores_empty_results(self):
        """Test retrieve when no documents are found."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = []
            mock_get_store.return_value = mock_vector_store

            results = retrieve_with_scores("nonexistent query")

            assert results == []

    def test_retrieve_with_scores_score_range(self):
        """Test that scores are in valid range [0, 1]."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_doc = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = [
                (mock_doc, 0.95),
            ]
            mock_get_store.return_value = mock_vector_store

            results = retrieve_with_scores("query")

            assert 0 <= results[0][1] <= 1

    def test_retrieve_with_scores_custom_collection(self):
        """Test retrieve with custom collection name."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.return_value = []
            mock_get_store.return_value = mock_vector_store

            collection_name = "custom_docs"
            retrieve_with_scores("query", collection_name=collection_name)

            mock_get_store.assert_called_with(collection_name)

    def test_retrieve_with_scores_error_handling(self):
        """Test retrieve error handling."""
        with patch("rag.retriever.get_vector_store") as mock_get_store:
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_relevance_scores.side_effect = Exception(
                "Vector store error"
            )
            mock_get_store.return_value = mock_vector_store

            with pytest.raises(Exception):
                retrieve_with_scores("query")
