"""
Generator node: Synthesizes answer from graded document chunks.

Uses relevant chunks as context to generate a comprehensive answer
with inline citations in the format [TS XX.XXX §Y.Z].
"""

import re
from typing import TYPE_CHECKING

from specagent.llm import create_llm

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


GENERATOR_PROMPT = """You are a 3GPP specification expert assistant. Your task is to extract precise answers from 3GPP specification context.

Question: {question}

Context (from 3GPP specifications):
---
{context}
---

Instructions - Follow these steps:

STEP 1 - SEARCH: Review each numbered chunk to identify which contain information relevant to the question.
- Look for exact parameter names, values, units, and technical terms
- Note chunk numbers that contain relevant information

STEP 2 - EXTRACT: Extract the specific answer from the relevant chunks.
- For numerical parameters: Extract the exact value WITH units (e.g., "160 ms", "25 m", "9 dB")
- For technical terms: Extract exact terminology as stated in the specification
- For descriptive answers: Synthesize information from multiple chunks if needed

STEP 3 - VERIFY: Cross-check your answer against the original chunks.
- Ensure the answer is directly stated or clearly implied in the context
- Verify units and numerical values are correct
- Confirm you haven't added information not present in the chunks

STEP 4 - CITE: Add inline citations for every claim.
- REQUIRED: Every statement MUST have a citation in format [TS XX.XXX §Y.Z]
- Use the exact source references from the chunk headers
- If information comes from multiple chunks, cite all relevant sources

STEP 5 - RESPOND: Provide your final answer.
- Start with the direct answer to the question
- Include supporting details if helpful
- ALL citations MUST be inline using [TS XX.XXX §Y.Z] format

CRITICAL RULES:
✓ Answer ONLY from the provided context - NO external knowledge
✓ ALWAYS cite sources - every claim needs [TS XX.XXX §Y.Z]
✓ Extract exact values with units for numerical parameters
✓ Say "I don't have enough information" ONLY if:
  - The question asks for information completely absent from all chunks
  - The chunks discuss related but different topics
  - You cannot find the specific parameter, value, or term requested

✗ DO NOT say "insufficient information" if:
  - The answer is present but requires reading multiple chunks
  - The information is stated in technical terminology
  - The value is embedded in a table or formula

Answer:"""


# Regex pattern to extract citations in format: [TS XX.XXX §Y.Z]
# Captures: TS number (with optional dash) and section number (with dots/letters)
CITATION_PATTERN = re.compile(
    r'\[TS\s+(\d+\.\d+(?:-\d+)?)\s+§\s*([0-9A-Za-z.]+)\]'
)


def generator_node(state: "GraphState") -> "GraphState":
    """
    Generate answer from graded chunks.

    Args:
        state: Current graph state with graded_chunks containing relevant docs

    Returns:
        Updated state with generation and citations populated
    """
    # Import at runtime to avoid circular imports
    from specagent.graph.state import Citation  # noqa: PLC0415

    # Get question and graded chunks from state
    question = state.get("question", "")
    graded_chunks = state.get("graded_chunks", [])

    # Filter for relevant chunks only
    relevant_chunks = [
        gc.chunk for gc in graded_chunks if gc.relevant == "yes"
    ]

    # Handle case where no relevant chunks are available
    if not relevant_chunks:
        state["generation"] = (
            "I don't have enough information in the available specifications "
            "to fully answer this question."
        )
        state["citations"] = []
        return state

    try:
        # Format chunks into context string with source metadata and numbering
        context_parts = []
        for idx, chunk in enumerate(relevant_chunks, start=1):
            # Format: **Chunk N** [TS XX.XXX §Y.Z]: content
            source_ref = f"[TS {chunk.spec_id.replace('TS', '').replace('.', '.', 1)} §{chunk.section}]"
            context_parts.append(f"**Chunk {idx}** {source_ref}:\n{chunk.content}")

        context = "\n\n".join(context_parts)

        # Initialize LLM (auto-selects based on config)
        llm = create_llm()

        # Format prompt with question and context
        prompt = GENERATOR_PROMPT.format(
            question=question,
            context=context
        )

        # Call LLM to generate answer
        generation = llm.invoke(prompt)

        # Convert to string and strip whitespace
        generation = generation.strip() if isinstance(generation, str) else str(generation).strip()

        # Parse citations from the generated response
        citations = []
        for match in CITATION_PATTERN.finditer(generation):
            spec_num = match.group(1)  # e.g., "38.321" or "38.101-1"
            section = match.group(2)   # e.g., "5.4" or "5.3.7.1"
            raw_citation = match.group(0)  # Full match like "[TS 38.321 §5.4]"

            # Normalize spec_id (TS38.321 format - no spaces)
            spec_id = f"TS{spec_num.replace(' ', '')}"

            # Create Citation object
            citation = Citation(
                spec_id=spec_id,
                section=section,
                raw_citation=raw_citation,
                chunk_preview=""  # Optional field
            )
            citations.append(citation)

        # Update state
        state["generation"] = generation
        state["citations"] = citations

    except Exception as e:
        # Handle errors gracefully
        state["error"] = f"Generator error: {e!s}"
        state["generation"] = None
        state["citations"] = []

    return state
