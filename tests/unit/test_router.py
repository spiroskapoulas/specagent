"""Unit tests for router node."""

import pytest
from unittest.mock import MagicMock, patch

from specagent.graph.state import GraphState, create_initial_state
from specagent.nodes.router import RouteDecision, router_node


@pytest.mark.unit
class TestRouteDecision:
    """Tests for RouteDecision Pydantic model."""

    def test_route_decision_retrieve(self):
        """Test creating RouteDecision for retrieve."""
        decision = RouteDecision(
            route="retrieve",
            reasoning="Question is about 3GPP specifications"
        )

        assert decision.route == "retrieve"
        assert decision.reasoning == "Question is about 3GPP specifications"

    def test_route_decision_reject(self):
        """Test creating RouteDecision for reject."""
        decision = RouteDecision(
            route="reject",
            reasoning="Question is about cooking, not telecom"
        )

        assert decision.route == "reject"
        assert decision.reasoning == "Question is about cooking, not telecom"

    def test_route_decision_validation(self):
        """Test that invalid route values are rejected."""
        with pytest.raises(ValueError):
            RouteDecision(route="invalid", reasoning="test")


@pytest.mark.unit
class TestRouterNode:
    """Tests for router_node function."""

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_retrieve_decision(self, mock_hf_hub):
        """Test router node with retrieve decision."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="This question is about 5G NR HARQ processes"
        )
        mock_hf_hub.return_value = mock_llm

        # Create initial state
        state = create_initial_state("What is the maximum number of HARQ processes in NR?")

        # Call router node
        result = router_node(state)

        # Verify state was updated
        assert result["route_decision"] == "retrieve"
        assert result["route_reasoning"] == "This question is about 5G NR HARQ processes"
        assert result["question"] == "What is the maximum number of HARQ processes in NR?"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_reject_decision(self, mock_hf_hub):
        """Test router node with reject decision."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="This question is about cooking, not telecommunications"
        )
        mock_hf_hub.return_value = mock_llm

        # Create initial state
        state = create_initial_state("What is the best recipe for chocolate cake?")

        # Call router node
        result = router_node(state)

        # Verify state was updated
        assert result["route_decision"] == "reject"
        assert result["route_reasoning"] == "This question is about cooking, not telecommunications"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_with_3gpp_question(self, mock_hf_hub):
        """Test router correctly routes 3GPP-related questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Question about RRC procedures in 3GPP TS 38.331"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("What triggers RRC connection re-establishment?")
        result = router_node(state)

        assert result["route_decision"] == "retrieve"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_with_5g_question(self, mock_hf_hub):
        """Test router correctly routes 5G questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Question about 5G carrier aggregation"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("How many component carriers can be aggregated in 5G?")
        result = router_node(state)

        assert result["route_decision"] == "retrieve"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_with_off_topic_question(self, mock_hf_hub):
        """Test router correctly rejects off-topic questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="Question is about sports, not telecommunications"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Who won the World Cup in 2022?")
        result = router_node(state)

        assert result["route_decision"] == "reject"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_preserves_other_state_fields(self, mock_hf_hub):
        """Test that router only modifies routing fields."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Test reasoning"
        )
        mock_hf_hub.return_value = mock_llm

        # Create state with additional fields
        state = create_initial_state("Test question")
        state["rewrite_count"] = 1
        state["error"] = None

        result = router_node(state)

        # Verify other fields are preserved
        assert result["rewrite_count"] == 1
        assert result["error"] is None
        assert result["question"] == "Test question"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_llm_call_format(self, mock_hf_hub):
        """Test that LLM is called with correct format."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Test"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        router_node(state)

        # Verify with_structured_output was called with RouteDecision
        mock_llm.with_structured_output.assert_called_once_with(RouteDecision)

        # Verify invoke was called with prompt containing question
        invoke_call_args = mock_structured.invoke.call_args
        prompt = invoke_call_args[0][0]
        assert "Test question" in prompt

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_handles_llm_error(self, mock_hf_hub):
        """Test router handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM API error")
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")

        # Should handle error and populate error field
        result = router_node(state)

        assert "error" in result
        assert result["error"] is not None
        # Should default to reject on error for safety
        assert result["route_decision"] == "reject"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_uses_hf_hub_settings(self, mock_hf_hub):
        """Test that router uses HuggingFaceHub with correct settings."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Test"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        router_node(state)

        # Verify HuggingFaceHub was initialized
        mock_hf_hub.assert_called_once()

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_with_telecom_terminology(self, mock_hf_hub):
        """Test router recognizes various telecom terminology."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Question contains telecom terminology"
        )
        mock_hf_hub.return_value = mock_llm

        # Test various telecom terms
        telecom_questions = [
            "What is PDCCH?",
            "Explain gNB-DU and gNB-CU",
            "How does carrier aggregation work?",
            "What is the F1 interface?",
        ]

        for question in telecom_questions:
            state = create_initial_state(question)
            result = router_node(state)
            assert result["route_decision"] == "retrieve", f"Failed for: {question}"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_empty_question(self, mock_hf_hub):
        """Test router handles empty questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="Empty question"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("")
        result = router_node(state)

        assert result["route_decision"] == "reject"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_whitespace_only_question(self, mock_hf_hub):
        """Test router handles whitespace-only questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="Whitespace only"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("   \n\t  ")
        result = router_node(state)

        assert result["route_decision"] == "reject"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_very_long_question(self, mock_hf_hub):
        """Test router handles very long questions."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Long but valid 3GPP question"
        )
        mock_hf_hub.return_value = mock_llm

        # Create a very long question
        long_question = "What is the HARQ process? " * 100
        state = create_initial_state(long_question)
        result = router_node(state)

        assert result["route_decision"] == "retrieve"
        assert result["question"] == long_question

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_unicode_question(self, mock_hf_hub):
        """Test router handles questions with unicode characters."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Valid question with unicode"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("What is 5G's maximum throughput in Gbps? 🚀")
        result = router_node(state)

        assert result["route_decision"] == "retrieve"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_special_characters(self, mock_hf_hub):
        """Test router handles questions with special characters."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Valid question with special chars"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("What is [TS 38.321 §5.4] about?")
        result = router_node(state)

        assert result["route_decision"] == "retrieve"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_mixed_content_question(self, mock_hf_hub):
        """Test router with mixed telecom and non-telecom content."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Contains telecom content despite mixed topics"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state(
            "I was cooking dinner when I wondered about 5G NR HARQ processes"
        )
        result = router_node(state)

        assert result["route_decision"] == "retrieve"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_error_sets_correct_fields(self, mock_hf_hub):
        """Test router error handling sets all expected fields."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception(
            "API rate limit exceeded"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        result = router_node(state)

        # Verify all error-related fields
        assert result["route_decision"] == "reject"
        assert result["route_reasoning"] == "Error occurred during routing"
        assert "error" in result
        assert "Router error" in result["error"]
        assert "API rate limit exceeded" in result["error"]

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_initialization_error(self, mock_hf_hub):
        """Test router handles LLM initialization errors."""
        mock_hf_hub.side_effect = Exception("Failed to initialize HuggingFaceHub")

        state = create_initial_state("Test question")
        result = router_node(state)

        assert result["route_decision"] == "reject"
        assert "error" in result

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_structured_output_error(self, mock_hf_hub):
        """Test router handles structured output parsing errors."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = Exception("Structured output error")
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        result = router_node(state)

        assert result["route_decision"] == "reject"
        assert "error" in result

    @patch('specagent.nodes.router.HuggingFaceHub')
    @patch('specagent.nodes.router.settings')
    def test_router_uses_settings_correctly(self, mock_settings, mock_hf_hub):
        """Test router uses configuration from settings."""
        # Mock settings
        mock_settings.llm_model = "test-model"
        mock_settings.hf_api_key_value = "test-key"
        mock_settings.llm_temperature = 0.2
        mock_settings.llm_max_tokens = 512

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Test"
        )
        mock_hf_hub.return_value = mock_llm

        state = create_initial_state("Test question")
        router_node(state)

        # Verify HuggingFaceHub was called with correct settings
        mock_hf_hub.assert_called_once()
        call_kwargs = mock_hf_hub.call_args[1]
        assert call_kwargs["repo_id"] == "test-model"
        assert call_kwargs["huggingfacehub_api_token"] == "test-key"
        assert call_kwargs["model_kwargs"]["temperature"] == 0.2
        assert call_kwargs["model_kwargs"]["max_new_tokens"] == 512

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_prompt_contains_question(self, mock_hf_hub):
        """Test that router prompt includes the user's question."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="Test"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_hf_hub.return_value = mock_llm

        test_question = "What is the maximum bandwidth for 5G NR?"
        state = create_initial_state(test_question)
        router_node(state)

        # Get the prompt that was passed to invoke
        invoke_call_args = mock_structured.invoke.call_args
        prompt = invoke_call_args[0][0]

        # Verify question is in prompt
        assert test_question in prompt
        assert "3GPP" in prompt or "router" in prompt.lower()

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_state_not_modified_on_error(self, mock_hf_hub):
        """Test that original question is preserved even on error."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("Error")
        mock_hf_hub.return_value = mock_llm

        original_question = "Original question"
        state = create_initial_state(original_question)
        state["rewrite_count"] = 0

        result = router_node(state)

        # Original fields should be preserved
        assert result["question"] == original_question
        assert result["rewrite_count"] == 0

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_multiple_calls_independent(self, mock_hf_hub):
        """Test that multiple router calls don't interfere with each other."""
        mock_llm = MagicMock()

        # First call returns retrieve
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="retrieve",
            reasoning="First call"
        )
        mock_hf_hub.return_value = mock_llm

        state1 = create_initial_state("Question 1")
        result1 = router_node(state1)

        # Second call returns reject
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="Second call"
        )

        state2 = create_initial_state("Question 2")
        result2 = router_node(state2)

        # Verify each call is independent
        assert result1["route_decision"] == "retrieve"
        assert result1["route_reasoning"] == "First call"
        assert result2["route_decision"] == "reject"
        assert result2["route_reasoning"] == "Second call"

    @patch('specagent.nodes.router.HuggingFaceHub')
    def test_router_missing_question_key(self, mock_hf_hub):
        """Test router handles state without question key."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
            route="reject",
            reasoning="No question provided"
        )
        mock_hf_hub.return_value = mock_llm

        # Create state without question
        state: GraphState = GraphState()
        result = router_node(state)

        # Should handle gracefully with empty string
        assert "route_decision" in result
