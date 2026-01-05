"""
Hallucination checker node: Verifies generated answer is grounded in sources.

Uses LLM-as-judge to compare the generated answer against source chunks
and identify any claims not supported by the retrieved context.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from specagent.llm import create_llm

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


class HallucinationResult(BaseModel):
    """Structured output for hallucination checking."""

    grounded: Literal["yes", "no", "partial"] = Field(
        description="Whether all claims in the answer are supported by sources"
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="List of claims not found in the source documents"
    )


HALLUCINATION_PROMPT = """You are a fact-checker for a 3GPP specification assistant.
Your job is to verify that every factual claim in the generated answer is supported
by the source documents.

Source documents:
---
{sources}
---

Generated answer:
---
{answer}
---

Carefully verify each factual claim in the answer:
1. Is every technical detail (numbers, parameters, procedures) found in the sources?
2. Are any claims made that go beyond what the sources state?
3. Are any specifications or section numbers cited that don't match the sources?

Respond with ONLY a JSON object in this exact format:
{{"grounded": "yes", "ungrounded_claims": []}}
or
{{"grounded": "partial", "ungrounded_claims": ["claim 1", "claim 2"]}}
or
{{"grounded": "no", "ungrounded_claims": ["claim 1", "claim 2", "claim 3"]}}

Where grounded is "yes" if all claims are fully supported, "partial" if most claims are supported,
or "no" if significant claims are not supported."""


def hallucination_check_node(state: "GraphState") -> "GraphState":
    """
    Check if generated answer is grounded in source documents.

    Args:
        state: Current graph state with generation and graded_chunks

    Returns:
        Updated state with hallucination_check result
    """
    # Get generation and graded chunks from state
    generation = state.get("generation")
    graded_chunks = state.get("graded_chunks", [])

    # Handle case where generation is None or empty
    if not generation or generation.strip() == "":
        state["hallucination_check"] = "grounded"
        state["ungrounded_claims"] = []
        return state

    # Get relevant chunks only
    relevant_chunks = [
        gc.chunk for gc in graded_chunks if gc.relevant == "yes"
    ]

    # Handle case where no relevant chunks are available
    # If there's a generation but no sources, it's likely ungrounded
    if not relevant_chunks:
        try:
            # Initialize LLM (auto-selects based on config)
            llm = create_llm()

            # Format prompt with empty sources
            prompt = HALLUCINATION_PROMPT.format(
                sources="(No source documents provided)",
                answer=generation
            )

            # Call LLM to check for hallucinations
            response = llm.invoke(prompt)

            # Parse JSON response
            import json
            import re

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                check_result = HallucinationResult(**parsed)
            else:
                check_result = HallucinationResult(**json.loads(response))

            # Map HallucinationResult.grounded to state hallucination_check values
            if check_result.grounded == "yes":
                state["hallucination_check"] = "grounded"
            elif check_result.grounded == "no":
                state["hallucination_check"] = "not_grounded"
            else:  # partial
                state["hallucination_check"] = "partial"

            state["ungrounded_claims"] = check_result.ungrounded_claims

        except Exception as e:
            # Handle errors gracefully
            state["error"] = f"Hallucination check error: {e!s}"
            state["hallucination_check"] = "grounded"
            state["ungrounded_claims"] = []

        return state

    try:
        # Format chunks into sources string
        source_parts = []
        for chunk in relevant_chunks:
            # Format: [TS XX.XXX §Y.Z]: content
            source_ref = f"[TS {chunk.spec_id.replace('TS', '').replace('.', '.', 1)} §{chunk.section}]"
            source_parts.append(f"{source_ref}: {chunk.content}")

        sources = "\n\n".join(source_parts)

        # Initialize LLM (auto-selects based on config)
        llm = create_llm()

        # Format prompt with sources and answer
        prompt = HALLUCINATION_PROMPT.format(
            sources=sources,
            answer=generation
        )

        # Call LLM to check for hallucinations
        response = llm.invoke(prompt)

        # Parse JSON response
        import json
        import re

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            result = HallucinationResult(**parsed)
        else:
            result = HallucinationResult(**json.loads(response))

        # Map HallucinationResult.grounded to state hallucination_check values
        if result.grounded == "yes":
            state["hallucination_check"] = "grounded"
        elif result.grounded == "no":
            state["hallucination_check"] = "not_grounded"
        else:  # partial
            state["hallucination_check"] = "partial"

        state["ungrounded_claims"] = result.ungrounded_claims

    except Exception as e:
        # Handle errors gracefully
        state["error"] = f"Hallucination check error: {e!s}"
        state["hallucination_check"] = "grounded"
        state["ungrounded_claims"] = []

    return state
