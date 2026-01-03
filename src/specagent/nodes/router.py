"""
Router node: Determines if a query relates to 3GPP specifications.

Routes queries to either:
    - "retrieve": Query is about 3GPP/telecom, proceed with retrieval
    - "reject": Query is off-topic, return polite rejection

Uses structured output from LLM to get routing decision with reasoning.
"""

from typing import TYPE_CHECKING, Literal

from langchain_community.llms import HuggingFaceHub
from pydantic import BaseModel, Field

from specagent.config import settings

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""

    route: Literal["retrieve", "reject"] = Field(
        description="'retrieve' for 3GPP questions, 'reject' for off-topic"
    )
    reasoning: str = Field(
        description="Brief explanation of the routing decision"
    )


ROUTER_PROMPT = """You are a router for a 3GPP specification assistant.
Determine if the following question relates to 3GPP/telecom standards.

Question: {question}

If the question is about 3GPP specifications, 5G, LTE, NR, RAN, core network,
telecommunications protocols, or any telecom standards topic, route to "retrieve".

If the question is completely unrelated (e.g., cooking, sports, general knowledge
unrelated to telecom), route to "reject".

Respond with your routing decision and brief reasoning."""


def router_node(state: "GraphState") -> "GraphState":
    """
    Route query to retrieval or rejection.

    Args:
        state: Current graph state containing the user's question

    Returns:
        Updated state with route_decision set to "retrieve" or "reject"
    """
    # Get question from state
    question = state.get("question", "")

    try:
        # Initialize HuggingFace LLM
        llm = HuggingFaceHub(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.hf_api_key_value,
            model_kwargs={
                "temperature": settings.llm_temperature,
                "max_new_tokens": settings.llm_max_tokens,
            },
        )

        # Get structured output
        structured_llm = llm.with_structured_output(RouteDecision)

        # Format prompt with question
        prompt = ROUTER_PROMPT.format(question=question)

        # Call LLM
        decision: RouteDecision = structured_llm.invoke(prompt)

        # Update state with decision
        state["route_decision"] = decision.route
        state["route_reasoning"] = decision.reasoning

    except Exception as e:
        # Handle errors gracefully - default to reject for safety
        state["route_decision"] = "reject"
        state["route_reasoning"] = "Error occurred during routing"
        state["error"] = f"Router error: {str(e)}"

    return state
