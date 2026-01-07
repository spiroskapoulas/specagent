"""
Unit tests for the retriever node.

Tests the retriever node's ability to:
    - Embed queries using HuggingFaceEmbedder
    - Search FAISS index for similar chunks
    - Handle errors gracefully
    - Update graph state correctly
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from specagent.graph.state import GraphState, RetrievedChunk, create_initial_state


@pytest.mark.unit
def test_retriever_node_with_query(sample_chunks, sample_embeddings, tmp_index_dir):
    """Test retriever node successfully retrieves chunks for a query."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.indexer import FAISSIndex

    # Build real index for testing
    real_index = FAISSIndex()
    real_index.build(sample_chunks, sample_embeddings)

    # Create state with a question
    state: GraphState = create_initial_state("What is HARQ in NR?")

    # Mock the embedder and index getters
    mock_query_embedding = sample_embeddings[0:1]  # aembed_texts returns array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = real_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify state was updated with retrieved chunks
    assert "retrieved_chunks" in result_state
    assert len(result_state["retrieved_chunks"]) > 0
    assert len(result_state["retrieved_chunks"]) <= 10  # Top-10 limit

    # Verify chunks have correct structure
    for chunk in result_state["retrieved_chunks"]:
        assert isinstance(chunk, RetrievedChunk)
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.spec_id, str)
        assert isinstance(chunk.section, str)
        assert isinstance(chunk.similarity_score, float)
        assert 0.0 <= chunk.similarity_score <= 1.0
        assert isinstance(chunk.chunk_id, str)

    # Verify embedder was called with the query
    mock_embedder.aembed_texts.assert_called_once_with(["What is HARQ in NR?"])

    # Verify no errors
    assert result_state.get("error") is None


@pytest.mark.unit
def test_retriever_node_with_rewritten_question(
    sample_chunks, sample_embeddings, tmp_index_dir
):
    """Test retriever uses rewritten_question if available."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.indexer import FAISSIndex

    # Build real index for testing
    real_index = FAISSIndex()
    real_index.build(sample_chunks, sample_embeddings)

    # Create state with both question and rewritten_question
    state: GraphState = create_initial_state("HARQ in NR?")
    state["rewritten_question"] = "What is the HARQ process in 5G NR Release 18?"

    # Mock the embedder and index getters
    mock_query_embedding = sample_embeddings[0:1]  # aembed_texts returns array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = real_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify embedder was called with rewritten_question, not original
    mock_embedder.aembed_texts.assert_called_once_with(
        ["What is the HARQ process in 5G NR Release 18?"]
    )

    # Verify chunks were retrieved
    assert len(result_state["retrieved_chunks"]) > 0


@pytest.mark.unit
def test_retriever_node_index_not_loaded(sample_chunks, sample_embeddings):
    """Test retriever handles missing index gracefully."""
    from specagent.nodes.retriever import retriever_node

    # Create state
    state: GraphState = create_initial_state("What is HARQ?")

    # Mock embedder
    mock_query_embedding = sample_embeddings[0:1]  # aembed_texts returns array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            # Mock get_faiss_index to raise FileNotFoundError
            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.side_effect = FileNotFoundError("Index file not found: /nonexistent/path/faiss.index.index")

                # Run retriever node
                result_state = retriever_node(state)

    # Verify error was set
    assert "error" in result_state
    assert result_state["error"] is not None
    assert "index" in result_state["error"].lower() or "not found" in result_state["error"].lower()

    # Verify retrieved_chunks is empty
    assert result_state.get("retrieved_chunks", []) == []


@pytest.mark.unit
def test_retriever_node_embedding_error(tmp_index_dir, sample_chunks, sample_embeddings):
    """Test retriever handles embedding errors gracefully."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.indexer import FAISSIndex

    # Build real index for testing
    real_index = FAISSIndex()
    real_index.build(sample_chunks, sample_embeddings)

    # Create state
    state: GraphState = create_initial_state("What is HARQ?")

    # Mock embedder to raise error
    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(side_effect=Exception("API rate limit exceeded"))
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = real_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify error was set
    assert "error" in result_state
    assert result_state["error"] is not None
    assert "retriever" in result_state["error"].lower() or "embedding" in result_state["error"].lower()

    # Verify retrieved_chunks is empty
    assert result_state.get("retrieved_chunks", []) == []


@pytest.mark.unit
def test_retriever_node_returns_top_10(sample_embeddings, tmp_index_dir):
    """Test retriever returns at most 10 chunks."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.chunker import Chunk
    from specagent.retrieval.indexer import FAISSIndex

    # Create 20 sample chunks
    many_chunks = [
        Chunk(
            content=f"This is chunk {i} about 3GPP specifications.",
            metadata={
                "source_file": f"TS38.{i}.md",
                "section_header": f"Section {i}",
                "chunk_index": i,
            },
        )
        for i in range(20)
    ]

    # Create embeddings for 20 chunks
    rng = np.random.default_rng(42)
    many_embeddings = rng.random((20, 384)).astype(np.float32)
    norms = np.linalg.norm(many_embeddings, axis=1, keepdims=True)
    many_embeddings = many_embeddings / norms

    # Build real index for testing
    real_index = FAISSIndex()
    real_index.build(many_chunks, many_embeddings)

    # Create state
    state: GraphState = create_initial_state("Tell me about 3GPP")

    # Mock embedder and index getters
    mock_query_embedding = many_embeddings[0:1]  # aembed_texts returns array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = real_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify exactly 10 chunks returned
    assert len(result_state["retrieved_chunks"]) == 10

    # Verify scores are in descending order
    scores = [chunk.similarity_score for chunk in result_state["retrieved_chunks"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_retriever_node_chunk_metadata_conversion(
    sample_chunks, sample_embeddings, tmp_index_dir
):
    """Test that Chunk metadata is correctly converted to RetrievedChunk fields."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.indexer import FAISSIndex

    # Build the real index for testing
    real_index = FAISSIndex()
    real_index.build(sample_chunks, sample_embeddings)

    # Create state
    state: GraphState = create_initial_state("What is HARQ?")

    # Mock embedder and index getters
    mock_query_embedding = sample_embeddings[0:1]  # aembed_texts returns array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = real_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify chunk metadata was correctly converted
    first_chunk = result_state["retrieved_chunks"][0]

    # Should have extracted spec_id from source_file
    assert first_chunk.source_file in ["TS38.321.md", "TS38.101-1.md", "TS38.331.md", "TS38.211.md", "TS38.401.md"]

    # Should have section from section_header
    assert first_chunk.section != ""

    # Should have a chunk_id
    assert first_chunk.chunk_id != ""

    # Should have the content
    assert len(first_chunk.content) > 0


@pytest.mark.unit
def test_retriever_node_empty_index(tmp_index_dir):
    """Test retriever handles empty index gracefully."""
    from specagent.nodes.retriever import retriever_node
    from specagent.retrieval.indexer import FAISSIndex

    # Create empty index
    empty_index = FAISSIndex()
    empty_index.build([], np.empty((0, 384), dtype=np.float32))

    # Create state
    state: GraphState = create_initial_state("What is HARQ?")

    # Mock embedder and index getters
    mock_query_embedding = np.random.rand(1, 384).astype(np.float32)  # aembed_texts returns 2D array

    with patch("specagent.config.settings.use_local_embeddings", False):
        with patch("specagent.nodes.retriever.get_hf_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder.aembed_texts = AsyncMock(return_value=mock_query_embedding)
            mock_get_embedder.return_value = mock_embedder

            with patch("specagent.nodes.retriever.get_faiss_index") as mock_get_index:
                mock_get_index.return_value = empty_index

                # Run retriever node
                result_state = retriever_node(state)

    # Verify empty results (not an error)
    assert result_state.get("retrieved_chunks", []) == []
    # Should not have an error since empty index is valid
    assert result_state.get("error") is None
