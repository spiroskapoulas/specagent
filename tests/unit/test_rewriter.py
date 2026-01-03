"""Unit tests for rewriter node."""

from unittest.mock import MagicMock, patch

import pytest

from specagent.graph.state import GraphState, RetrievedChunk, create_initial_state
from specagent.nodes.rewriter import rewriter_node


@pytest.mark.unit
class TestRewriterNode:
    """Tests for rewriter_node function."""

    def _create_state_with_low_confidence(
        self,
        question: str,
        rewrite_count: int = 0,
        chunks_data: list[dict] | None = None
    ) -> GraphState:
        """Helper to create state with low confidence graded chunks."""
        state = create_initial_state(question)
        state["route_decision"] = "retrieve"
        state["rewrite_count"] = rewrite_count
        state["average_confidence"] = 0.4  # Below threshold

        if chunks_data is None:
            chunks_data = [
                {"content": "The PDCCH carries downlink control information."},
                {"content": "Component carriers can be aggregated."},
            ]

        state["retrieved_chunks"] = [
            RetrievedChunk(
                content=chunk["content"],
                spec_id=chunk.get("spec_id", "TS38.211"),
                section=chunk.get("section", "7.3"),
                similarity_score=chunk.get("similarity_score", 0.5),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                source_file=chunk.get("source_file", "TS38.211.md")
            )
            for i, chunk in enumerate(chunks_data)
        ]
        return state

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_successful_rewrite(self, mock_hf_hub):
        """Test rewriter successfully rewrites query."""
        # Mock LLM to return rewritten question
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "What is the maximum number of HARQ processes in 5G NR Release 18?"
        mock_hf_hub.return_value = mock_llm

        # Create state with low confidence (needs rewriting)
        state = self._create_state_with_low_confidence(
            "What is the max HARQ processes?",
            rewrite_count=0
        )

        # Call rewriter node
        result = rewriter_node(state)

        # Verify rewritten question is set
        assert result["rewritten_question"] == "What is the maximum number of HARQ processes in 5G NR Release 18?"

        # Verify rewrite count is incremented
        assert result["rewrite_count"] == 1

        # Verify LLM was called
        mock_llm.invoke.assert_called_once()

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_max_rewrites_reached(self, mock_hf_hub):
        """Test rewriter stops when max rewrites limit is reached."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten question"
        mock_hf_hub.return_value = mock_llm

        # Create state with rewrite_count at max (default is 2)
        state = self._create_state_with_low_confidence(
            "What is HARQ?",
            rewrite_count=2
        )

        # Call rewriter node
        result = rewriter_node(state)

        # Verify no rewriting happened
        assert result["rewritten_question"] is None
        assert result["rewrite_count"] == 2  # Unchanged

        # Verify LLM was NOT called
        mock_llm.invoke.assert_not_called()

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_increments_count_correctly(self, mock_hf_hub):
        """Test rewriter increments rewrite count from non-zero."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Improved question about HARQ processes"
        mock_hf_hub.return_value = mock_llm

        # Start with rewrite_count=1
        state = self._create_state_with_low_confidence(
            "HARQ processes?",
            rewrite_count=1
        )

        result = rewriter_node(state)

        # Should increment to 2
        assert result["rewrite_count"] == 2
        assert result["rewritten_question"] == "Improved question about HARQ processes"

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_uses_original_question(self, mock_hf_hub):
        """Test that rewriter uses original question in prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_hf_hub.return_value = mock_llm

        question = "What are the UE capabilities for carrier aggregation?"
        state = self._create_state_with_low_confidence(question)

        rewriter_node(state)

        # Verify the prompt contains the original question
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert question in prompt

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_includes_chunk_context(self, mock_hf_hub):
        """Test that rewriter includes retrieved chunks in prompt."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_hf_hub.return_value = mock_llm

        chunks = [
            {"content": "The PDCCH carries DCI information."},
            {"content": "Carrier aggregation allows multiple carriers."},
        ]
        state = self._create_state_with_low_confidence(
            "What is carrier aggregation?",
            chunks_data=chunks
        )

        rewriter_node(state)

        # Verify the prompt contains chunk summaries
        invoke_call_args = mock_llm.invoke.call_args
        prompt = invoke_call_args[0][0]
        # Should contain some reference to retrieved chunks
        assert "PDCCH" in prompt or "Carrier aggregation" in prompt

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_handles_llm_error(self, mock_hf_hub):
        """Test rewriter handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_hf_hub.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")

        result = rewriter_node(state)

        # Should populate error field
        assert "error" in result
        assert result["error"] is not None
        assert "error" in result["error"].lower()

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_preserves_other_state_fields(self, mock_hf_hub):
        """Test that rewriter only modifies rewrite fields."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten question"
        mock_hf_hub.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")
        state["route_reasoning"] = "Test routing"
        state["average_confidence"] = 0.3

        result = rewriter_node(state)

        # Verify other fields are preserved
        assert result["question"] == "Test question"
        assert result["route_reasoning"] == "Test routing"
        assert result["average_confidence"] == 0.3

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_with_empty_chunks(self, mock_hf_hub):
        """Test rewriter handles empty retrieved chunks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten with no context"
        mock_hf_hub.return_value = mock_llm

        state = self._create_state_with_low_confidence(
            "Test question",
            chunks_data=[]
        )

        result = rewriter_node(state)

        # Should still rewrite, just without chunk context
        assert result["rewritten_question"] == "Rewritten with no context"
        assert result["rewrite_count"] == 1

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_limit_minus_one(self, mock_hf_hub):
        """Test rewriter allows rewrite when count is one below limit."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Final rewrite"
        mock_hf_hub.return_value = mock_llm

        # At count=1, should allow one more rewrite (max is 2)
        state = self._create_state_with_low_confidence(
            "Test question",
            rewrite_count=1
        )

        result = rewriter_node(state)

        # Should allow this rewrite
        assert result["rewritten_question"] == "Final rewrite"
        assert result["rewrite_count"] == 2

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_strips_whitespace(self, mock_hf_hub):
        """Test rewriter strips extra whitespace from LLM output."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  \n  What is the maximum number of HARQ processes?  \n  "
        mock_hf_hub.return_value = mock_llm

        state = self._create_state_with_low_confidence("What is HARQ?")

        result = rewriter_node(state)

        # Should strip whitespace
        assert result["rewritten_question"] == "What is the maximum number of HARQ processes?"

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_uses_correct_settings(self, mock_hf_hub):
        """Test that rewriter uses correct LLM settings from config."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_hf_hub.return_value = mock_llm

        state = self._create_state_with_low_confidence("Test question")

        rewriter_node(state)

        # Verify HuggingFaceHub was initialized with correct parameters
        init_call = mock_hf_hub.call_args
        assert "repo_id" in init_call[1] or len(init_call[0]) > 0
        assert "huggingfacehub_api_token" in init_call[1]
        assert "model_kwargs" in init_call[1]

    @patch('specagent.nodes.rewriter.HuggingFaceHub')
    def test_rewriter_prompt_format(self, mock_hf_hub):
        """Test that rewriter prompt contains all necessary components."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Rewritten"
        mock_hf_hub.return_value = mock_llm

        question = "What is HARQ?"
        chunks = [{"content": "HARQ is a retransmission protocol."}]
        state = self._create_state_with_low_confidence(question, chunks_data=chunks)

        rewriter_node(state)

        prompt = mock_llm.invoke.call_args[0][0]

        # Verify prompt structure
        assert question in prompt
        assert "HARQ" in prompt  # From chunk content
        assert any(keyword in prompt.lower() for keyword in ["rewrite", "reformulate", "specific"])
