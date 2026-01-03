"""
Grader node: Scores relevance of retrieved chunks to the query.

For each retrieved chunk, determines:
    - relevant: "yes" or "no"
    - confidence: 0.0 to 1.0

If average confidence is below threshold, triggers query rewriting.
"""

from typing import TYPE_CHECKING, Literal

from langchain_community.llms import HuggingFaceHub
from pydantic import BaseModel, Field

from specagent.config import settings

if TYPE_CHECKING:
    from specagent.graph.state import GraphState


class GradeResult(BaseModel):
    """Structured output for document grading."""

    relevant: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to answering the question"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the relevance assessment"
    )


GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document chunk 
to a user question about 3GPP telecommunications specifications.

Question: {question}

Document chunk:
---
{document}
---

Does this document contain information relevant to answering the question?
Consider: exact matches, related concepts, prerequisite information.

Assess the relevance and your confidence in that assessment."""


def grader_node(state: "GraphState") -> "GraphState":
    """
    Grade retrieved chunks for relevance to the query.

    Args:
        state: Current graph state with retrieved_chunks populated

    Returns:
        Updated state with graded_chunks containing relevance scores
    """
    from specagent.graph.state import GradedChunk

    # Get question and retrieved chunks from state
    question = state.get("question", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Handle empty chunks case
    if not retrieved_chunks:
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0
        return state

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
        structured_llm = llm.with_structured_output(GradeResult)

        # Grade each chunk
        graded_chunks = []
        total_confidence = 0.0

        for chunk in retrieved_chunks:
            # Format prompt with question and chunk content
            prompt = GRADER_PROMPT.format(
                question=question,
                document=chunk.content
            )

            # Call LLM to grade the chunk
            grade: GradeResult = structured_llm.invoke(prompt)

            # Create GradedChunk
            graded_chunk = GradedChunk(
                chunk=chunk,
                relevant=grade.relevant,
                confidence=grade.confidence
            )
            graded_chunks.append(graded_chunk)

            # Accumulate confidence for average
            total_confidence += grade.confidence

        # Calculate average confidence
        average_confidence = total_confidence / len(graded_chunks)

        # Update state
        state["graded_chunks"] = graded_chunks
        state["average_confidence"] = average_confidence

    except Exception as e:
        # Handle errors gracefully
        state["error"] = f"Grader error: {str(e)}"
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0

    return state
