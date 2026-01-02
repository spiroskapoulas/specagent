# =============================================================================
# 3GPP SpecAgent Dockerfile
# Multi-stage build for minimal production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install production dependencies only
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir ".[eval]"

# Export installed packages list
RUN pip freeze > /requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="3GPP SpecAgent"
LABEL org.opencontainers.image.description="Agentic RAG for 3GPP specifications"
LABEL org.opencontainers.image.version="0.1.0"

# Create non-root user
RUN groupadd --gid 1000 specagent \
    && useradd --uid 1000 --gid specagent --shell /bin/bash --create-home specagent

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /requirements.txt /app/requirements.txt

# Create data directories
RUN mkdir -p /app/data/index /app/data/raw /app/data/processed \
    && chown -R specagent:specagent /app

# Switch to non-root user
USER specagent

# Environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default configuration
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    ENABLE_TRACING=false

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "specagent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
