"""
Unit tests for graph state module.
"""

import pytest

from specagent.graph.state import (
    Citation,
    GradedChunk,
    GraphState,
    RetrievedChunk,
    create_initial_state,
)


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_retrieved_chunk_creation(self):
        """RetrievedChunk should store all required fields."""
        chunk = RetrievedChunk(
            content="Test content",
            spec_id="TS38.321",
            section="5.4",
            similarity_score=0.95,
            chunk_id="TS38.321.md:0",
            source_file="TS38.321.md",
        )

        assert chunk.content == "Test content"
        assert chunk.spec_id == "TS38.321"
        assert chunk.section == "5.4"
        assert chunk.similarity_score == 0.95
        assert chunk.chunk_id == "TS38.321.md:0"

    def test_retrieved_chunk_default_source_file(self):
        """source_file should default to empty string."""
        chunk = RetrievedChunk(
            content="Test",
            spec_id="TS38.321",
            section="5.4",
            similarity_score=0.9,
            chunk_id="test",
        )

        assert chunk.source_file == ""


class TestGradedChunk:
    """Tests for GradedChunk dataclass."""

    def test_graded_chunk_creation(self, sample_chunks):
        """GradedChunk should wrap RetrievedChunk with grade info."""
        retrieved = RetrievedChunk(
            content=sample_chunks[0].content,
            spec_id=sample_chunks[0].spec_id,
            section=sample_chunks[0].section,
            similarity_score=0.9,
            chunk_id=sample_chunks[0].chunk_id,
        )

        graded = GradedChunk(
            chunk=retrieved,
            relevant="yes",
            confidence=0.85,
        )

        assert graded.chunk.content == sample_chunks[0].content
        assert graded.relevant == "yes"
        assert graded.confidence == 0.85


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Citation should store spec reference info."""
        citation = Citation(
            spec_id="TS38.321",
            section="5.4.1",
            raw_citation="[TS 38.321 §5.4.1]",
            chunk_preview="The UE shall support a maximum of 16 HARQ processes...",
        )

        assert citation.spec_id == "TS38.321"
        assert citation.section == "5.4.1"
        assert citation.raw_citation == "[TS 38.321 §5.4.1]"


class TestGraphState:
    """Tests for GraphState TypedDict."""

    def test_create_initial_state(self, sample_question):
        """create_initial_state should set defaults correctly."""
        state = create_initial_state(sample_question)

        assert state["question"] == sample_question
        assert state["rewritten_question"] is None
        assert state["retrieved_chunks"] == []
        assert state["graded_chunks"] == []
        assert state["citations"] == []
        assert state["rewrite_count"] == 0
        assert state["generation"] is None
        assert state["error"] is None

    def test_graph_state_is_mutable(self, sample_question):
        """GraphState should allow modification of fields."""
        state = create_initial_state(sample_question)

        state["route_decision"] = "retrieve"
        state["rewrite_count"] = 1
        state["generation"] = "Test answer"

        assert state["route_decision"] == "retrieve"
        assert state["rewrite_count"] == 1
        assert state["generation"] == "Test answer"

    def test_graph_state_allows_partial(self):
        """GraphState should allow partial initialization (total=False)."""
        # This should not raise an error
        state: GraphState = {
            "question": "Test question",
        }

        assert state["question"] == "Test question"
        # Other fields should not be present
        assert "generation" not in state
