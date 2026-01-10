"""
Graph state definition for the agentic RAG pipeline.

The GraphState TypedDict defines all data that flows through the LangGraph
workflow. Each node reads from and writes to this shared state.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict


@dataclass
class RetrievedChunk:
    """A document chunk retrieved from the FAISS index."""

    content: str
    """The text content of the chunk."""

    spec_id: str
    """Specification identifier (e.g., 'TS38.331')."""

    section: str
    """Section reference within the spec (e.g., '5.3.3')."""

    similarity_score: float
    """Cosine similarity to the query (0.0 to 1.0)."""

    chunk_id: str
    """Unique identifier for this chunk."""

    source_file: str = ""
    """Original filename from TSpec-LLM dataset."""


@dataclass
class GradedChunk:
    """A retrieved chunk with relevance grading."""

    chunk: RetrievedChunk
    """The original retrieved chunk."""

    relevant: Literal["yes", "no"]
    """Whether the chunk is relevant to the question."""

    confidence: float
    """Confidence score for the relevance assessment (0.0 to 1.0)."""


@dataclass
class Citation:
    """A citation extracted from the generated answer."""

    spec_id: str
    """Specification identifier (e.g., 'TS38.331')."""

    section: str
    """Section reference (e.g., '5.3.3')."""

    raw_citation: str
    """The citation as it appears in the text (e.g., '[TS 38.331 §5.3.3]')."""

    chunk_preview: str = ""
    """Preview of the source chunk content."""


class GraphState(TypedDict, total=False):
    """
    State schema for the agentic RAG workflow.

    All fields are optional (total=False) to allow incremental population
    as the workflow progresses through nodes.

    Flow:
        1. User provides question
        2. Router decides: retrieve or reject
        3. Retriever fetches chunks (populates retrieved_chunks)
        4. Grader scores chunks (populates graded_chunks)
        5. If scores low: Rewriter reformulates (updates rewritten_question)
        6. Generator creates answer (populates generation, citations)
        7. Hallucination checker verifies (populates hallucination_check)
    """

    # ==========================================================================
    # Input
    # ==========================================================================
    question: str
    """The original user question."""

    # ==========================================================================
    # Routing
    # ==========================================================================
    route_decision: Literal["retrieve", "reject"]
    """Router's decision on how to handle the query."""

    route_reasoning: str
    """Router's explanation for the routing decision."""

    # ==========================================================================
    # Retrieval
    # ==========================================================================
    rewritten_question: Optional[str]
    """Reformulated question for improved retrieval (if rewriting was needed)."""

    retrieved_chunks: list[RetrievedChunk]
    """Chunks retrieved from FAISS index."""

    # ==========================================================================
    # Grading
    # ==========================================================================
    graded_chunks: list[GradedChunk]
    """Retrieved chunks with relevance scores."""

    average_confidence: float
    """Average confidence score across graded chunks."""

    # ==========================================================================
    # Rewriting
    # ==========================================================================
    rewrite_count: int
    """Number of query rewrites performed (max: settings.max_rewrites)."""

    # ==========================================================================
    # Generation
    # ==========================================================================
    generation: Optional[str]
    """The generated answer text."""

    citations: list[Citation]
    """Citations extracted from the generated answer."""

    # ==========================================================================
    # Hallucination Check
    # ==========================================================================
    hallucination_check: Literal["grounded", "not_grounded", "partial"]
    """Result of hallucination verification."""

    ungrounded_claims: list[str]
    """List of claims not supported by source documents."""

    # ==========================================================================
    # Metadata
    # ==========================================================================
    error: Optional[str]
    """Error message if something went wrong."""

    processing_time_ms: float
    """Total processing time in milliseconds."""

    # ==========================================================================
    # Timing Breakdown
    # ==========================================================================
    node_timings: dict[str, float]
    """Timing for each node execution in milliseconds (e.g., {'router': 150.2, 'retriever': 320.5})."""

    llm_inference_times: list[dict[str, float]]
    """LLM inference timings with context (e.g., [{'node': 'router', 'inference_ms': 145.3}])."""


def create_initial_state(question: str) -> GraphState:
    """
    Create an initial graph state from a user question.

    Args:
        question: The user's natural language question

    Returns:
        GraphState with question populated and defaults set
    """
    return GraphState(
        question=question,
        rewritten_question=None,
        retrieved_chunks=[],
        graded_chunks=[],
        citations=[],
        rewrite_count=0,
        generation=None,
        error=None,
        ungrounded_claims=[],
        node_timings={},
        llm_inference_times=[],
    )
