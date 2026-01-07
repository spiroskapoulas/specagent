"""
LangGraph workflow construction and execution.

Defines the agentic RAG graph with conditional routing:
    - Router decides retrieve vs reject
    - Grader triggers rewriting if confidence is low
    - Hallucination checker can trigger regeneration

Graph visualization can be exported via get_graph_visualization().
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from specagent.config import settings
from specagent.graph.state import GraphState, create_initial_state
from specagent.nodes import (
    generator_node,
    grader_node,
    hallucination_check_node,
    retriever_node,
    rewriter_node,
    router_node,
)


def should_retrieve(state: GraphState) -> Literal["retrieve", "reject"]:
    """
    Conditional edge: Route based on router decision.

    Returns:
        "retrieve" to continue with retrieval, "reject" to end
    """
    return state.get("route_decision", "reject")


def should_rewrite(state: GraphState) -> Literal["rewrite", "generate"]:
    """
    Conditional edge: Decide if query needs rewriting.

    Uses a fast similarity heuristic first, then checks quality metrics:
        1. If top-3 chunks have high similarity (>= threshold), skip rewriting
        2. Otherwise, rewrite if:
           - Average confidence is below threshold OR relevant chunk percentage is below threshold
           - Haven't exceeded max rewrites

    Returns:
        "rewrite" to reformulate query, "generate" to proceed
    """
    rewrite_count = state.get("rewrite_count", 0)
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Fast heuristic: Skip rewriting if top-3 chunks have high similarity
    if retrieved_chunks:
        top_3_chunks = retrieved_chunks[:3]
        avg_similarity = sum(chunk.similarity_score for chunk in top_3_chunks) / len(top_3_chunks)

        if avg_similarity >= settings.high_similarity_threshold:
            # High-quality retrieval, skip rewriting even if grader is uncertain
            return "generate"

    # Check quality metrics from grader
    avg_confidence = state.get("average_confidence", 0.0)
    graded_chunks = state.get("graded_chunks", [])

    # Calculate percentage of relevant chunks (only if grader has run)
    if graded_chunks:
        relevant_count = sum(1 for chunk in graded_chunks if chunk.relevant == "yes")
        relevant_percentage = relevant_count / len(graded_chunks)

        # Rewrite if quality is poor AND we haven't exceeded max rewrites
        quality_is_poor = (
            avg_confidence < settings.grader_confidence_threshold
            or relevant_percentage < settings.min_relevant_chunk_percentage
        )
    else:
        # No graded chunks yet - fall back to confidence-only check
        quality_is_poor = avg_confidence < settings.grader_confidence_threshold

    if quality_is_poor and rewrite_count < settings.max_rewrites:
        return "rewrite"

    return "generate"


def should_regenerate(state: GraphState) -> Literal["regenerate", "finish"]:
    """
    Conditional edge: Decide if answer needs regeneration.

    Triggers regeneration if hallucination check failed.
    Only allows one regeneration attempt.

    Returns:
        "regenerate" to try again, "finish" to complete
    """
    hallucination_result = state.get("hallucination_check", "grounded")

    # Only regenerate once to avoid infinite loops
    # TODO: Track regeneration attempts in state
    if hallucination_result == "not_grounded":
        return "regenerate"

    return "finish"


def build_graph() -> CompiledStateGraph:
    """
    Build and compile the agentic RAG graph.

    Graph structure:
        START
          │
          ▼
        [router]
          │
          ├── reject ──────────────────────────────────► END
          │
          └── retrieve
                │
                ▼
            [retriever]
                │
                ▼
            [grader]
                │
                ├── rewrite ──► [rewriter] ──► [retriever] (loop)
                │
                └── generate
                      │
                      ▼
                  [generator]
                      │
                      ▼
              [hallucination_check]
                      │
                      ├── regenerate ──► [generator] (retry once)
                      │
                      └── finish ──────────────────────► END

    Returns:
        Compiled LangGraph ready for execution
    """
    # Initialize graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # Add edges from START
    workflow.add_edge(START, "router")

    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retriever",
            "reject": END,
        },
    )

    # Retriever always goes to grader
    workflow.add_edge("retriever", "grader")

    # Grader conditional edges
    workflow.add_conditional_edges(
        "grader",
        should_rewrite,
        {
            "rewrite": "rewriter",
            "generate": "generator",
        },
    )

    # Rewriter loops back to retriever
    workflow.add_edge("rewriter", "retriever")

    # Generator goes to hallucination check
    workflow.add_edge("generator", "hallucination_check")

    # Hallucination check conditional edges
    workflow.add_conditional_edges(
        "hallucination_check",
        should_regenerate,
        {
            "regenerate": "generator",
            "finish": END,
        },
    )

    # Compile the graph
    return workflow.compile()


def run_query(question: str) -> GraphState:
    """
    Execute a query through the agentic RAG pipeline.

    Args:
        question: User's natural language question

    Returns:
        Final graph state with answer, citations, and metadata
    """
    import time

    # Create initial state
    state = create_initial_state(question)

    # Build and run graph
    graph = build_graph()

    start_time = time.perf_counter()
    final_state = graph.invoke(state)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Add timing metadata
    final_state["processing_time_ms"] = elapsed_ms

    return final_state


def get_graph_visualization() -> str:
    """
    Generate Mermaid diagram of the workflow.

    Returns:
        Mermaid diagram string for visualization

    Example:
        >>> mermaid = get_graph_visualization()
        >>> print(mermaid)  # Paste into mermaid.live
    """
    graph = build_graph()
    return graph.get_graph().draw_mermaid()


def save_graph_image(path: str = "docs/architecture.png") -> None:
    """
    Save graph visualization as PNG image.

    Requires graphviz to be installed.

    Args:
        path: Output path for the PNG file
    """
    graph = build_graph()
    graph.get_graph().draw_png(path)
