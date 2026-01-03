"""Unit tests for grader node."""

import pytest
from unittest.mock import MagicMock, patch

from specagent.graph.state import GraphState, RetrievedChunk, GradedChunk, create_initial_state
from specagent.nodes.grader import GradeResult, grader_node


@pytest.mark.unit
class TestGradeResult:
    """Tests for GradeResult Pydantic model."""

    def test_grade_result_relevant(self):
        """Test creating GradeResult for relevant chunk."""
        result = GradeResult(
            relevant="yes",
            confidence=0.85
        )

        assert result.relevant == "yes"
        assert result.confidence == 0.85

    def test_grade_result_not_relevant(self):
        """Test creating GradeResult for irrelevant chunk."""
        result = GradeResult(
            relevant="no",
            confidence=0.95
        )

        assert result.relevant == "no"
        assert result.confidence == 0.95

    def test_grade_result_validation_relevant(self):
        """Test that invalid relevant values are rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="maybe", confidence=0.5)

    def test_grade_result_validation_confidence_low(self):
        """Test that confidence below 0 is rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="yes", confidence=-0.1)

    def test_grade_result_validation_confidence_high(self):
        """Test that confidence above 1 is rejected."""
        with pytest.raises(ValueError):
            GradeResult(relevant="yes", confidence=1.5)

    def test_grade_result_confidence_boundaries(self):
        """Test that confidence at boundaries 0 and 1 is valid."""
        result_min = GradeResult(relevant="yes", confidence=0.0)
        result_max = GradeResult(relevant="no", confidence=1.0)

        assert result_min.confidence == 0.0
        assert result_max.confidence == 1.0


@pytest.mark.unit
class TestGraderNode:
    """Tests for grader_node function."""

    def _create_state_with_chunks(self, question: str, chunks_data: list[dict]) -> GraphState:
        """Helper to create state with retrieved chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"
        state["retrieved_chunks"] = [
            RetrievedChunk(
                content=chunk["content"],
                spec_id=chunk.get("spec_id", "TS38.321"),
                section=chunk.get("section", "5.4"),
                similarity_score=chunk.get("similarity_score", 0.8),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                source_file=chunk.get("source_file", "TS38.321.md")
            )
            for i, chunk in enumerate(chunks_data)
        ]
        return state

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_all_relevant(self, mock_hf_hub):
        """Test grader node with all relevant chunks."""
        # Mock LLM to return all relevant
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            GradeResult(relevant="yes", confidence=0.9),
            GradeResult(relevant="yes", confidence=0.85),
            GradeResult(relevant="yes", confidence=0.8),
        ]
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with 3 chunks
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16."},
            {"content": "HARQ processes are used for retransmission handling."},
            {"content": "Both FDD and TDD support 16 HARQ processes."},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?",
            chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify all chunks are graded
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "yes" for gc in result["graded_chunks"])

        # Verify average confidence
        expected_avg = (0.9 + 0.85 + 0.8) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_all_irrelevant(self, mock_hf_hub):
        """Test grader node with all irrelevant chunks."""
        # Mock LLM to return all irrelevant
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            GradeResult(relevant="no", confidence=0.95),
            GradeResult(relevant="no", confidence=0.9),
            GradeResult(relevant="no", confidence=0.85),
        ]
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with irrelevant chunks
        chunks = [
            {"content": "The PDCCH carries downlink control information."},
            {"content": "Component carriers can be aggregated."},
            {"content": "The F1 interface connects gNB-DU and gNB-CU."},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?",
            chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify all chunks are graded as irrelevant
        assert len(result["graded_chunks"]) == 3
        assert all(gc.relevant == "no" for gc in result["graded_chunks"])

        # Verify average confidence
        expected_avg = (0.95 + 0.9 + 0.85) / 3
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_mixed_relevance(self, mock_hf_hub):
        """Test grader node with mixed relevant and irrelevant chunks."""
        # Mock LLM to return mixed results
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            GradeResult(relevant="yes", confidence=0.9),
            GradeResult(relevant="no", confidence=0.8),
            GradeResult(relevant="yes", confidence=0.85),
            GradeResult(relevant="no", confidence=0.75),
        ]
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with mixed chunks
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16."},
            {"content": "The PDCCH carries downlink control information."},
            {"content": "HARQ processes handle retransmissions in NR."},
            {"content": "The F1 interface is used in split architecture."},
        ]
        state = self._create_state_with_chunks(
            "What is the maximum number of HARQ processes in NR?",
            chunks
        )

        # Call grader node
        result = grader_node(state)

        # Verify mixed results
        assert len(result["graded_chunks"]) == 4
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["graded_chunks"][1].relevant == "no"
        assert result["graded_chunks"][2].relevant == "yes"
        assert result["graded_chunks"][3].relevant == "no"

        # Verify average confidence
        expected_avg = (0.9 + 0.8 + 0.85 + 0.75) / 4
        assert result["average_confidence"] == pytest.approx(expected_avg)

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_preserves_chunk_data(self, mock_hf_hub):
        """Test that grader preserves original chunk data in GradedChunk."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeResult(relevant="yes", confidence=0.85)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [
            {
                "content": "Test content",
                "spec_id": "TS38.321",
                "section": "5.4.1",
                "similarity_score": 0.92,
                "chunk_id": "test_chunk_123",
                "source_file": "TS38.321.md"
            }
        ]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Verify original chunk data is preserved
        graded_chunk = result["graded_chunks"][0]
        assert graded_chunk.chunk.content == "Test content"
        assert graded_chunk.chunk.spec_id == "TS38.321"
        assert graded_chunk.chunk.section == "5.4.1"
        assert graded_chunk.chunk.similarity_score == 0.92
        assert graded_chunk.chunk.chunk_id == "test_chunk_123"
        assert graded_chunk.chunk.source_file == "TS38.321.md"

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_single_chunk(self, mock_hf_hub):
        """Test grader with single chunk."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeResult(relevant="yes", confidence=0.88)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Single test chunk"}]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        assert len(result["graded_chunks"]) == 1
        assert result["graded_chunks"][0].relevant == "yes"
        assert result["average_confidence"] == 0.88

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_empty_chunks(self, mock_hf_hub):
        """Test grader with no retrieved chunks."""
        mock_llm = MagicMock()
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["retrieved_chunks"] = []

        result = grader_node(state)

        # Should return empty graded_chunks and 0 average confidence
        assert result["graded_chunks"] == []
        assert result["average_confidence"] == 0.0

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_llm_call_format(self, mock_hf_hub):
        """Test that LLM is called with correct format."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeResult(relevant="yes", confidence=0.8)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Test chunk content"}]
        state = self._create_state_with_chunks("What is HARQ?", chunks)

        grader_node(state)

        # Verify with_structured_output was called with GradeResult
        mock_llm.with_structured_output.assert_called_once_with(GradeResult)

        # Verify invoke was called with prompt containing question and chunk
        invoke_call_args = mock_structured.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "What is HARQ?" in prompt
        assert "Test chunk content" in prompt

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_handles_llm_error(self, mock_hf_hub):
        """Test grader handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM API error")
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Test chunk"}]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_preserves_other_state_fields(self, mock_hf_hub):
        """Test that grader only modifies grading fields."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeResult(relevant="yes", confidence=0.8)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Test chunk"}]
        state = self._create_state_with_chunks("Test question", chunks)
        state["route_reasoning"] = "Test routing"
        state["rewrite_count"] = 1

        result = grader_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test question"
        assert result["route_reasoning"] == "Test routing"
        assert result["rewrite_count"] == 1

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_multiple_chunks_different_specs(self, mock_hf_hub):
        """Test grader with chunks from different specifications."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            GradeResult(relevant="yes", confidence=0.9),
            GradeResult(relevant="yes", confidence=0.85),
            GradeResult(relevant="no", confidence=0.7),
        ]
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [
            {"content": "HARQ info", "spec_id": "TS38.321", "section": "5.4"},
            {"content": "More HARQ", "spec_id": "TS38.331", "section": "7.2"},
            {"content": "Unrelated", "spec_id": "TS38.401", "section": "6.1"},
        ]
        state = self._create_state_with_chunks("What is HARQ?", chunks)

        result = grader_node(state)

        # Verify chunks from different specs are graded correctly
        assert len(result["graded_chunks"]) == 3
        assert result["graded_chunks"][0].chunk.spec_id == "TS38.321"
        assert result["graded_chunks"][1].chunk.spec_id == "TS38.331"
        assert result["graded_chunks"][2].chunk.spec_id == "TS38.401"

    @patch('specagent.nodes.grader.HuggingFaceHub')
    def test_grader_average_confidence_calculation(self, mock_hf_hub):
        """Test that average confidence is calculated correctly."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        # Test with known values for easy verification
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        mock_structured.invoke.side_effect = [
            GradeResult(relevant="yes", confidence=c) for c in confidences
        ]
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": f"Chunk {i}"} for i in range(5)]
        state = self._create_state_with_chunks("Test question", chunks)

        result = grader_node(state)

        # Verify average is correct: (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5 = 0.7
        expected_avg = sum(confidences) / len(confidences)
        assert result["average_confidence"] == pytest.approx(expected_avg)
