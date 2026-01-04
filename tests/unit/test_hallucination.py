"""Unit tests for hallucination checker node."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GradedChunk, GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.hallucination import HallucinationResult, hallucination_check_node


@pytest.mark.unit
class TestHallucinationResult:
    """Tests for HallucinationResult Pydantic model."""

    def test_hallucination_result_grounded(self):
        """Test creating HallucinationResult for fully grounded answer."""
        result = HallucinationResult(grounded="yes", ungrounded_claims=[])

        assert result.grounded == "yes"
        assert result.ungrounded_claims == []

    def test_hallucination_result_not_grounded(self):
        """Test creating HallucinationResult for ungrounded answer."""
        result = HallucinationResult(
            grounded="no",
            ungrounded_claims=[
                "The UE supports 32 HARQ processes",
                "TDD requires special handling",
            ],
        )

        assert result.grounded == "no"
        assert len(result.ungrounded_claims) == 2
        assert "32 HARQ processes" in result.ungrounded_claims[0]

    def test_hallucination_result_partial(self):
        """Test creating HallucinationResult for partially grounded answer."""
        result = HallucinationResult(
            grounded="partial", ungrounded_claims=["Some minor detail not in sources"]
        )

        assert result.grounded == "partial"
        assert len(result.ungrounded_claims) == 1

    def test_hallucination_result_validation_grounded(self):
        """Test that invalid grounded values are rejected."""
        with pytest.raises(ValueError):
            HallucinationResult(grounded="maybe", ungrounded_claims=[])

    def test_hallucination_result_default_claims(self):
        """Test that ungrounded_claims defaults to empty list."""
        result = HallucinationResult(grounded="yes")
        assert result.ungrounded_claims == []


@pytest.mark.unit
class TestHallucinationCheckNode:
    """Tests for hallucination_check_node function."""

    def _create_state_with_generation(
        self, question: str, generation: str, chunks_data: list[dict]
    ) -> GraphState:
        """Helper to create state with generation and graded chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"

        # Create retrieved chunks
        retrieved_chunks = [
            RetrievedChunk(
                content=chunk["content"],
                spec_id=chunk.get("spec_id", "TS38.321"),
                section=chunk.get("section", "5.4"),
                similarity_score=chunk.get("similarity_score", 0.8),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                source_file=chunk.get("source_file", "TS38.321.md"),
            )
            for i, chunk in enumerate(chunks_data)
        ]
        state["retrieved_chunks"] = retrieved_chunks

        # Create graded chunks (mark all as relevant for simplicity)
        state["graded_chunks"] = [
            GradedChunk(chunk=chunk, relevant="yes", confidence=0.85) for chunk in retrieved_chunks
        ]

        state["generation"] = generation

        return state

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_fully_grounded(self, mock_hf_hub):
        """Test hallucination check with fully grounded answer."""
        # Mock LLM to return grounded result
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with generation that matches source
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
            {"content": "HARQ processes are used for retransmission handling in the MAC layer."},
        ]
        generation = "The maximum number of HARQ processes in NR is 16 for both FDD and TDD. [TS 38.321 §5.4]"

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify fully grounded
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_not_grounded(self, mock_hf_hub):
        """Test hallucination check with ungrounded answer."""
        # Mock LLM to return not grounded result
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="no",
            ungrounded_claims=[
                "The UE supports up to 32 HARQ processes",
                "TDD mode requires special configuration",
            ],
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with generation that contradicts sources
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
        ]
        generation = (
            "The UE supports up to 32 HARQ processes. TDD mode requires special configuration."
        )

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify not grounded
        assert result["hallucination_check"] == "not_grounded"
        assert len(result["ungrounded_claims"]) == 2
        assert "32 HARQ processes" in result["ungrounded_claims"][0]

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_partial(self, mock_hf_hub):
        """Test hallucination check with partially grounded answer."""
        # Mock LLM to return partial result
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="partial", ungrounded_claims=["Some implementation details may vary by vendor"]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        # Create state with mostly correct generation but some unsupported claims
        chunks = [
            {"content": "The maximum number of HARQ processes for NR is 16 for both FDD and TDD."},
        ]
        generation = "The maximum number of HARQ processes in NR is 16. Some implementation details may vary by vendor."

        state = self._create_state_with_generation(
            "What is the maximum number of HARQ processes in NR?", generation, chunks
        )

        # Call hallucination check node
        result = hallucination_check_node(state)

        # Verify partial grounding
        assert result["hallucination_check"] == "partial"
        assert len(result["ungrounded_claims"]) == 1
        assert "vendor" in result["ungrounded_claims"][0].lower()

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_no_generation(self, mock_hf_hub):
        """Test hallucination check when generation is None."""
        mock_llm = MagicMock()
        mock_hf_hub.return_value = mock_llm

        # Create state without generation
        state = create_initial_state("Test question")
        state["generation"] = None
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded (nothing to check)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_empty_generation(self, mock_hf_hub):
        """Test hallucination check with empty string generation."""
        mock_llm = MagicMock()
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = ""
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded (nothing to check)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_no_chunks_grounded(self, mock_hf_hub):
        """Test hallucination check with no chunks - grounded result."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as grounded
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_no_chunks_not_grounded(self, mock_hf_hub):
        """Test hallucination check with no chunks - not grounded result."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="no", ungrounded_claims=["All claims are ungrounded - no sources provided"]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as not grounded (no sources to verify against)
        assert result["hallucination_check"] == "not_grounded"
        assert len(result["ungrounded_claims"]) > 0

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_no_chunks_partial(self, mock_hf_hub):
        """Test hallucination check with no chunks - partial result."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="partial", ungrounded_claims=["Some claims are questionable"]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer without sources."
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should mark as partial
        assert result["hallucination_check"] == "partial"
        assert len(result["ungrounded_claims"]) > 0

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_no_chunks_llm_error(self, mock_hf_hub):
        """Test hallucination check handles LLM errors when no chunks."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM API error")
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        state["generation"] = "Some generated answer."
        state["graded_chunks"] = []

        result = hallucination_check_node(state)

        # Should populate error field and default to grounded
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_llm_call_format(self, mock_hf_hub):
        """Test that LLM is called with correct format."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "The maximum number of HARQ processes is 16."}]
        generation = "The answer is 16 HARQ processes."

        state = self._create_state_with_generation("Test?", generation, chunks)

        hallucination_check_node(state)

        # Verify with_structured_output was called with HallucinationResult
        mock_llm.with_structured_output.assert_called_once_with(HallucinationResult)

        # Verify invoke was called with prompt containing sources and answer
        invoke_call_args = mock_structured.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "The maximum number of HARQ processes is 16" in prompt
        assert "The answer is 16 HARQ processes" in prompt

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_handles_llm_error(self, mock_hf_hub):
        """Test hallucination check handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM API error")
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        state = self._create_state_with_generation("Test?", "Test answer", chunks)

        result = hallucination_check_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_preserves_other_state_fields(self, mock_hf_hub):
        """Test that hallucination check only modifies hallucination fields."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Test content"}]
        state = self._create_state_with_generation("Test?", "Test answer", chunks)
        state["route_reasoning"] = "Test routing"
        state["rewrite_count"] = 1
        state["average_confidence"] = 0.9

        result = hallucination_check_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test?"
        assert result["generation"] == "Test answer"
        assert result["route_reasoning"] == "Test routing"
        assert result["rewrite_count"] == 1
        assert result["average_confidence"] == 0.9

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_with_multiple_chunks(self, mock_hf_hub):
        """Test hallucination check with multiple source chunks."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [
            {"content": "HARQ processes: 16 maximum", "spec_id": "TS38.321", "section": "5.4"},
            {"content": "Both FDD and TDD supported", "spec_id": "TS38.321", "section": "5.4.1"},
            {"content": "MAC layer handles HARQ", "spec_id": "TS38.321", "section": "5.1"},
        ]
        generation = "NR supports 16 HARQ processes for both FDD and TDD, handled by MAC."

        state = self._create_state_with_generation("Test?", generation, chunks)

        hallucination_check_node(state)

        # Verify all chunks were included in the prompt
        invoke_call_args = mock_structured.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "HARQ processes: 16 maximum" in prompt
        assert "Both FDD and TDD supported" in prompt
        assert "MAC layer handles HARQ" in prompt

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_grounded_to_state_mapping(self, mock_hf_hub):
        """Test correct mapping from HallucinationResult.grounded to state values."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_hf_hub.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured

        chunks = [{"content": "Test content"}]

        # Test "yes" -> "grounded"
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        state = self._create_state_with_generation("Q?", "A", chunks)
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "grounded"

        # Test "no" -> "not_grounded"
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="no", ungrounded_claims=["claim"]
        )
        state = self._create_state_with_generation("Q?", "A", chunks)
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "not_grounded"

        # Test "partial" -> "partial"
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="partial", ungrounded_claims=["claim"]
        )
        state = self._create_state_with_generation("Q?", "A", chunks)
        result = hallucination_check_node(state)
        assert result["hallucination_check"] == "partial"

    @patch("specagent.nodes.hallucination.HuggingFaceHub")
    def test_hallucination_insufficient_info_response(self, mock_hf_hub):
        """Test hallucination check with 'I don't have enough information' response."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationResult(
            grounded="yes", ungrounded_claims=[]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        chunks = [{"content": "Some unrelated content"}]
        generation = "I don't have enough information in the available specifications to fully answer this question."

        state = self._create_state_with_generation("Complex question?", generation, chunks)

        result = hallucination_check_node(state)

        # Should be grounded (truthful acknowledgment of lack of info)
        assert result["hallucination_check"] == "grounded"
        assert result["ungrounded_claims"] == []
