"""
Generator node: Synthesizes answer from graded document chunks.

Uses relevant chunks as context to generate a comprehensive answer
with inline citations in the format [TS XX.XXX §Y.Z].
"""

import re
from typing import TYPE_CHECKING

from langchain_huggingface import HuggingFaceEndpoint

from specagent.config import settings

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


GENERATOR_PROMPT = """You are a 3GPP specification expert. Answer the question using ONLY
the provided context from official 3GPP documentation.

Question: {question}

Context (from 3GPP specifications):
---
{context}
---

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using format: [TS XX.XXX §Y.Z] where available
3. If the context doesn't contain enough information, say "I don't have enough information
   in the available specifications to fully answer this question."
4. Be precise and technical - this is for telecom engineers
5. If multiple specs are relevant, synthesize them coherently

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
        # Format chunks into context string with source metadata
        context_parts = []
        for chunk in relevant_chunks:
            # Format: [TS XX.XXX §Y.Z]: content
            source_ref = f"[TS {chunk.spec_id.replace('TS', '').replace('.', '.', 1)} §{chunk.section}]"
            context_parts.append(f"{source_ref}: {chunk.content}")

        context = "\n\n".join(context_parts)

        # Initialize HuggingFace LLM
        llm = HuggingFaceEndpoint(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.hf_api_key_value,
            temperature=settings.llm_temperature,
            max_new_tokens=settings.llm_max_tokens,
        )

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
