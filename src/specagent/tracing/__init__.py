"""
Observability and tracing with Arize Phoenix.

Provides OpenTelemetry-based tracing for the LangGraph pipeline
with automatic instrumentation of LangChain components.
"""

from specagent.tracing.phoenix import setup_tracing, traced

__all__ = ["setup_tracing", "traced"]
