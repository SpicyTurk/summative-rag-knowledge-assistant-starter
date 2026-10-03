from typing import Any, List

import chromadb
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    return chromadb.PersistentClient(path=Config.CHROMA_PATH)


def get_or_create_collection():

    client = get_chroma_client()
    return client.get_or_create_collection(name=Config.COLLECTION_NAME)


def get_embedding(text: str) -> list[float]:

    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": Config.EMBEDDING_MODEL,
            "input": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    embeddings = data.get("embeddings")
    if embeddings:
        return embeddings[0]

    if data.get("embedding"):
        return data["embedding"]

    raise ValueError("The embedding response did not contain an embedding.")


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """Add document chunks to the Chroma collection and return the count."""
    if not chunks:
        return 0

    collection = get_or_create_collection()

    ids = [chunk.id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]
    embeddings = [get_embedding(chunk.text) for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a user question."""
    if top_k is None:
        top_k = Config.TOP_K

    collection = get_or_create_collection()
    stored_count = collection.count()

    if stored_count == 0:
        return []

    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, stored_count),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", "Unknown Source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    return chunks