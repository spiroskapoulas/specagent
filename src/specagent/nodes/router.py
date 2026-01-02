"""
Router node: Determines if a query relates to 3GPP specifications.

Routes queries to either:
    - "retrieve": Query is about 3GPP/telecom, proceed with retrieval
    - "reject": Query is off-topic, return polite rejection

Uses structured output from LLM to get routing decision with reasoning.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

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
    # TODO: Implement router logic
    # 1. Get question from state
    # 2. Call LLM with structured output (RouteDecision)
    # 3. Update state["route_decision"]
    # 4. Return updated state
    raise NotImplementedError("Router node not yet implemented")
