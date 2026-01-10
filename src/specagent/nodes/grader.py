"""
Grader node: Scores relevance of retrieved chunks to the query.

For each retrieved chunk, determines:
    - relevant: "yes" or "no"
    - confidence: 0.0 to 1.0

If average confidence is below threshold, triggers query rewriting.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from specagent.llm import create_llm

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


class BatchGradeResult(BaseModel):
    """Structured output for batch document grading."""

    grades: list[GradeResult] = Field(
        description="List of grade results, one per document chunk in order"
    )


BATCH_GRADER_PROMPT = """You are a grader assessing relevance of retrieved document chunks
to a user question about 3GPP telecommunications specifications.

Question: {question}

You are given {num_chunks} document chunks to grade. For EACH chunk, determine if it contains
information relevant to answering the question. Consider: exact matches, related concepts, prerequisite information.

Document chunks:
{documents}

Respond with ONLY a JSON object with a "grades" array containing one grade per chunk IN THE SAME ORDER:
{{"grades": [
  {{"relevant": "yes", "confidence": 0.85}},
  {{"relevant": "no", "confidence": 0.2}},
  ...
]}}

You must provide exactly {num_chunks} grades. Each confidence must be between 0.0 and 1.0."""


def grader_node(state: "GraphState") -> "GraphState":
    """
    Grade retrieved chunks for relevance to the query.

    Grades only the top-3 chunks (by similarity score) for latency optimization.
    Retriever fetches 10 chunks, but we only grade the most promising ones.

    Args:
        state: Current graph state with retrieved_chunks populated

    Returns:
        Updated state with graded_chunks containing relevance scores
    """
    from specagent.graph.state import GradedChunk  # noqa: PLC0415

    # Get question and retrieved chunks from state
    question = state.get("question", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Only grade top-3 chunks for latency optimization
    chunks_to_grade = retrieved_chunks[:3]

    # Handle empty chunks case
    if not chunks_to_grade:
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0
        return state

    try:
        # Initialize LLM (auto-selects based on config)
        llm = create_llm()

        import json
        import re

        # Format chunks to grade into a single prompt
        documents_text = ""
        for i, chunk in enumerate(chunks_to_grade, 1):
            documents_text += f"\n--- Chunk {i} ---\n{chunk.content}\n"

        # Create batch grading prompt
        prompt = BATCH_GRADER_PROMPT.format(
            question=question,
            num_chunks=len(chunks_to_grade),
            documents=documents_text
        )

        # Single LLM call to grade all chunks at once
        response = llm.invoke(prompt)

        # Parse batch JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            batch_result = BatchGradeResult(**parsed)
        else:
            batch_result = BatchGradeResult(**json.loads(response))

        # Verify we got the right number of grades
        if len(batch_result.grades) != len(chunks_to_grade):
            raise ValueError(
                f"Expected {len(chunks_to_grade)} grades but got {len(batch_result.grades)}"
            )

        # Create GradedChunk objects from batch results
        graded_chunks = []
        total_confidence = 0.0

        for chunk, grade in zip(chunks_to_grade, batch_result.grades):
            graded_chunk = GradedChunk(
                chunk=chunk,
                relevant=grade.relevant,
                confidence=grade.confidence
            )
            graded_chunks.append(graded_chunk)
            total_confidence += grade.confidence

        # Calculate average confidence
        average_confidence = total_confidence / len(graded_chunks)

        # Update state
        state["graded_chunks"] = graded_chunks
        state["average_confidence"] = average_confidence

    except Exception as e:
        # Handle errors gracefully
        state["error"] = f"Grader error: {e!s}"
        state["graded_chunks"] = []
        state["average_confidence"] = 0.0

    return state
