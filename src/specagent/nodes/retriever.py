"""
Retriever node: Fetches relevant document chunks from FAISS index.

Embeds the user's query and searches the FAISS index for the top-10
most similar document chunks. Updates graph state with retrieved chunks.
"""

import asyncio
from typing import TYPE_CHECKING

from specagent.config import settings
from specagent.graph.state import RetrievedChunk
from specagent.retrieval.embeddings import HuggingFaceEmbedder, LocalEmbedder
from specagent.retrieval.indexer import FAISSIndex

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


def retriever_node(state: "GraphState") -> "GraphState":
    """
    Retrieve relevant chunks from FAISS index for the query.

    Args:
        state: Current graph state containing the user's question or rewritten_question

    Returns:
        Updated state with retrieved_chunks populated (top-10 similar chunks)
    """
    # Get query from state - prioritize rewritten_question over original question
    query = state.get("rewritten_question") or state.get("question", "")

    if not query:
        state["error"] = "Retriever error: No query found in state"
        state["retrieved_chunks"] = []
        return state

    try:
        # Initialize embedder based on config
        if settings.use_local_embeddings:
            embedder = LocalEmbedder()
            # Use synchronous method for local embeddings
            query_embedding = embedder.embed_query(query)
        else:
            embedder = HuggingFaceEmbedder()
            # Embed query asynchronously for HF API
            query_embedding = asyncio.run(embedder.aembed_texts([query]))[0]

        # Load FAISS index from disk
        index = FAISSIndex()
        index.load(settings.faiss_index_path)

        # Search index for top-10 similar chunks
        results = index.search(query_embedding, k=10)

        # Convert results to RetrievedChunk objects
        retrieved_chunks: list[RetrievedChunk] = []
        for chunk, similarity_score in results:
            # Extract spec_id from source_file (e.g., "TS38.321.md" -> "TS38.321")
            source_file = chunk.metadata.get("source_file", "unknown")
            spec_id = source_file.replace(".md", "").replace("-", ".")

            # Get section from metadata
            section = chunk.metadata.get("section_header", "")

            # Create unique chunk_id
            chunk_index = chunk.metadata.get("chunk_index", 0)
            chunk_id = f"{source_file}:{chunk_index}"

            # Create RetrievedChunk
            retrieved_chunk = RetrievedChunk(
                content=chunk.content,
                spec_id=spec_id,
                section=section,
                similarity_score=float(similarity_score),
                chunk_id=chunk_id,
                source_file=source_file,
            )
            retrieved_chunks.append(retrieved_chunk)

        # Update state with retrieved chunks
        state["retrieved_chunks"] = retrieved_chunks

    except FileNotFoundError as e:
        # Handle missing index file
        state["error"] = f"Retriever error: Index not found - {str(e)}"
        state["retrieved_chunks"] = []

    except Exception as e:
        # Handle other errors gracefully
        state["error"] = f"Retriever error: {str(e)}"
        state["retrieved_chunks"] = []

    return state
