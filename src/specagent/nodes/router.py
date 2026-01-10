"""
Router node: Determines if a query relates to 3GPP specifications.

Routes queries to either:
    - "retrieve": Query is about 3GPP/telecom, proceed with retrieval
    - "reject": Query is off-topic, return polite rejection

Uses structured output from LLM to get routing decision with reasoning.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from specagent.llm import create_llm

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


ROUTER_PROMPT = """You are a router for a 3GPP telecommunications specification assistant.
Your task is to determine if a question relates to 3GPP/telecom standards and should be routed to retrieval.

Question: {question}

=== IN-SCOPE TOPICS ===
Route to "retrieve" if the question is about ANY of the following 3GPP/telecom domains:

**Core Technologies:**
- 5G NR (New Radio), LTE, LTE-Advanced, 4G, 3G, 2G
- Radio Access Network (RAN), Core Network, evolved packet core (EPC), 5GC
- Protocols: RRC, PDCP, MAC, PHY, NAS, NGAP, Xn, F1, etc.

**Network Components:**
- Base stations: gNB, eNB, gNB-DU, gNB-CU, cell, sector
- User equipment (UE), mobile devices, terminals
- Network functions: AMF, SMF, UPF, PCF, etc.

**Non-Terrestrial Networks (NTN):**
- Satellite communications: LEO, MEO, GEO, geostationary, non-geostationary
- Aerial networks: UAV, drone, HAPS (High-Altitude Platform Station), aerial vehicles
- Satellite parameters: orbit, beam footprint, antenna aperture, Doppler shift, propagation delay
- Satellite terminals: VSAT (Very Small Aperture Terminal), satellite UE

**Technical Features:**
- Carrier aggregation, MIMO, beamforming, handover, RRC procedures
- Channel models, propagation, path loss, fading
- Spectrum, frequency bands, bandwidth
- QoS, throughput, latency, reliability
- Security, authentication, encryption

**Specifications & Standards:**
- Any question mentioning TS/TR numbers (e.g., TS 38.321, TR 36.777, TR 38.811)
- Technical parameters from 3GPP tables or sections
- Release-specific features (Rel-15, Rel-16, Rel-17, Rel-18, etc.)

=== OUT-OF-SCOPE TOPICS ===
Route to "reject" ONLY if the question is clearly about non-telecom domains:
- Cooking, recipes, food
- Sports, entertainment, movies, games
- General knowledge unrelated to telecommunications (history, geography, literature)
- Programming/coding questions not specific to telecom protocols
- General physics/math without telecom context

=== DECISION CRITERIA ===
**Route to "retrieve" if ANY of these are true:**
1. Question mentions 3GPP spec numbers (TS/TR followed by numbers)
2. Question contains telecom-specific terminology (see in-scope list above)
3. Question asks about technical parameters that could relate to telecom (even if domain is unclear)
4. You are uncertain whether it's telecom-related (default to retrieve for technical queries)

**Route to "reject" ONLY if ALL of these are true:**
1. Question contains zero telecom terminology
2. Question is clearly about a non-telecom domain (cooking, sports, etc.)
3. You are confident it cannot be answered using 3GPP specifications

=== EXAMPLES ===

**Retrieve Examples:**
- "What is the maximum number of HARQ processes in NR?" → retrieve (clear 5G question)
- "What factor impacts the Doppler shift for a geostationary satellite?" → retrieve (NTN/satellite topic)
- "For the LEO-600 orbit in Set-2 satellite parameters, what is the equivalent satellite antenna aperture for UL?" → retrieve (specific satellite parameter)
- "Explain RRC connection re-establishment" → retrieve (RRC protocol)
- "What is the BS antenna height in UMa scenario?" → retrieve (technical parameter)
- "How does handover work?" → retrieve (vague but clearly telecom)
- "What is PDCCH?" → retrieve (telecom acronym)

**Reject Examples:**
- "What is the best recipe for chocolate cake?" → reject (cooking)
- "Who won the World Cup in 2022?" → reject (sports)
- "How do I learn Python?" → reject (generic programming, not telecom-specific)
- "What is the capital of France?" → reject (general knowledge)

=== YOUR RESPONSE ===
Respond with ONLY a JSON object in this exact format:
{{"route": "retrieve", "reasoning": "your explanation"}}
or
{{"route": "reject", "reasoning": "your explanation"}}

**Remember**: When in doubt about technical questions, default to "retrieve". It's better to attempt retrieval than to incorrectly reject a valid 3GPP question."""


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
        # Initialize LLM (auto-selects based on config)
        llm = create_llm()

        # Format prompt with question
        prompt = ROUTER_PROMPT.format(question=question)

        # Call LLM
        response = llm.invoke(prompt)

        # Parse JSON response
        import json
        import re

        # Extract JSON from response (handle cases where LLM adds extra text)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            decision = RouteDecision(**parsed)
        else:
            # Fallback: try to parse the entire response
            decision = RouteDecision(**json.loads(response))

        # Update state with decision
        state["route_decision"] = decision.route
        state["route_reasoning"] = decision.reasoning

    except Exception as e:
        # Handle errors gracefully - default to reject for safety
        state["route_decision"] = "reject"
        state["route_reasoning"] = "Error occurred during routing"
        state["error"] = f"Router error: {str(e)}"

    return state
