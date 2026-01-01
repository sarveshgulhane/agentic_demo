import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import QDRANT_URL
from services import get_embeddings


def extract_pdf_text(pdf_path: str) -> list:
    """
    Extract text from PDF file using PyPDFLoader.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects with extracted text
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    return documents


def chunk_documents(
    documents: list, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def initialize_qdrant_collection(
    collection_name: str = "documents",
) -> QdrantClient:
    """
    Initialize Qdrant client and create/check collection.

    Args:
        collection_name: Name of the collection

    Returns:
        QdrantClient instance
    """
    client = QdrantClient(url=QDRANT_URL)

    # Check if collection exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        # Collection doesn't exist, create it
        embeddings = get_embeddings()
        embedding_dim = len(embeddings.embed_query("test"))

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim, distance=Distance.COSINE
            ),
        )
        print(
            f"Created collection '{collection_name}' with dimension {embedding_dim}."
        )

    return client


def ingest_pdf_to_qdrant(
    pdf_path: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> None:
    """
    Complete pipeline: Extract PDF text, chunk it, embed, and store in Qdrant.

    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the Qdrant collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print(f"Starting ingestion for: {pdf_path}")

    # Extract text from PDF
    print("Extracting text from PDF...")
    documents = extract_pdf_text(pdf_path)
    print(f"Extracted {len(documents)} pages from PDF.")

    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks from documents.")

    # Initialize Qdrant
    print("Initializing Qdrant collection...")
    initialize_qdrant_collection(collection_name)

    # Get embeddings and store in Qdrant
    print("Creating embeddings and storing in Qdrant...")
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
        prefer_grpc=False,
    )

    print(
        f"Successfully ingested {len(chunks)} chunks into Qdrant collection '{collection_name}'."
    )
    return vector_store


def ingest_directory(
    directory_path: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> None:
    """
    Ingest all PDF files from a directory.

    Args:
        directory_path: Path to directory containing PDFs
        collection_name: Name of the Qdrant collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    pdf_dir = Path(directory_path)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return

    print(f"Found {len(pdf_files)} PDF files to ingest.")

    for pdf_file in pdf_files:
        try:
            ingest_pdf_to_qdrant(
                str(pdf_file), collection_name, chunk_size, chunk_overlap
            )
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    documents_dir = "./documents"

    if os.path.exists(documents_dir):
        ingest_directory(documents_dir)
    else:
        print(f"Please add PDF files to the '{documents_dir}' directory")
