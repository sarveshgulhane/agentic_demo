from langchain_qdrant import QdrantVectorStore
from config import QDRANT_URL
from services.embedding_service import get_embeddings


def get_vector_store(collection_name: str = "documents") -> QdrantVectorStore:
    """
    Get QdrantVectorStore instance connected to the collection.
    
    Args:
        collection_name: Name of the Qdrant collection
        
    Returns:
        QdrantVectorStore instance
    """
    embeddings = get_embeddings()
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
        prefer_grpc=False
    )
    
    return vector_store


def retrieve_with_scores(query: str, k: int = 3, collection_name: str = "documents") -> list:
    """
    Retrieve relevant documents with similarity scores.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
        collection_name: Name of the Qdrant collection
        
    Returns:
        List of (document, score) tuples
    """
    vector_store = get_vector_store(collection_name)
    
    # Perform similarity search with relevance scores
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    return results

