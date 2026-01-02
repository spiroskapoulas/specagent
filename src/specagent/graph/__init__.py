"""
LangGraph workflow definition and state management.

Components:
    - state: TypedDict defining the graph state schema
    - workflow: Graph construction and compilation
"""

from specagent.graph.state import GraphState
from specagent.graph.workflow import build_graph, run_query

__all__ = [
    "GraphState",
    "build_graph",
    "run_query",
]
