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
