"""
FastAPI REST API for SpecAgent.

Endpoints:
    POST /query - Execute a query through the RAG pipeline
    GET /health - Health check for k8s probes
"""

from specagent.api.main import app, create_app

__all__ = ["app", "create_app"]
