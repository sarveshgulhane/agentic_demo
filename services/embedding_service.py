from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Embedding service for vector embeddings."""

    return HuggingFaceEmbeddings(
        model_name=model_name,
    )
