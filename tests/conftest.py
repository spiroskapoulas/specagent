"""
Pytest configuration and shared fixtures.

Provides common fixtures for:
    - Configuration with test values
    - Mock HuggingFace API responses
    - Sample document chunks
    - Temporary directories for indexes
"""

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_settings():
    """Provide test settings without requiring .env file."""
    with patch.dict(
        "os.environ",
        {
            "HF_API_KEY": "test-api-key",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "LLM_MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
            "CHUNK_SIZE": "512",
            "CHUNK_OVERLAP": "64",
            "ENABLE_TRACING": "false",
        },
    ):
        from specagent.config import Settings
        yield Settings()


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_chunks():
    """Provide sample 3GPP document chunks for testing."""
    from specagent.retrieval.chunker import Chunk

    return [
        Chunk(
            content="The maximum number of HARQ processes for NR is 16 for both FDD and TDD.",
            metadata={
                "source_file": "TS38.321.md",
                "section_header": "5.4 HARQ Entity",
                "chunk_index": 0,
            },
        ),
        Chunk(
            content="The UE shall support a maximum of 16 component carriers for carrier aggregation.",
            metadata={
                "source_file": "TS38.101-1.md",
                "section_header": "5.5A Carrier Aggregation",
                "chunk_index": 0,
            },
        ),
        Chunk(
            content="RRC connection re-establishment procedure is initiated when T311 expires.",
            metadata={
                "source_file": "TS38.331.md",
                "section_header": "5.3.7 RRC connection re-establishment",
                "chunk_index": 0,
            },
        ),
        Chunk(
            content="The PDCCH is used to carry downlink control information (DCI).",
            metadata={
                "source_file": "TS38.211.md",
                "section_header": "7.3 Physical downlink control channel",
                "chunk_index": 0,
            },
        ),
        Chunk(
            content="The gNB-DU and gNB-CU are connected via the F1 interface.",
            metadata={
                "source_file": "TS38.401.md",
                "section_header": "6.1 F1 Interface",
                "chunk_index": 0,
            },
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    # Generate random normalized embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.random((5, 384)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_question():
    """Provide a sample 3GPP-related question."""
    return "What is the maximum number of HARQ processes in NR?"


@pytest.fixture
def sample_off_topic_question():
    """Provide a sample off-topic question."""
    return "What is the best recipe for chocolate cake?"


# =============================================================================
# Mock API Fixtures
# =============================================================================

@pytest.fixture
def mock_hf_embedding_response():
    """Mock HuggingFace embedding API response."""
    def _mock_response(texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(hash(tuple(texts)) % 2**32)
        embeddings = rng.random((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return normalized.tolist()
    return _mock_response


@pytest.fixture
def mock_hf_llm_response():
    """Mock HuggingFace LLM API response."""
    def _mock_response(prompt: str) -> str:
        # Return structured responses based on prompt content
        if "router" in prompt.lower():
            return '{"route": "retrieve", "reasoning": "This is a 3GPP question"}'
        elif "grader" in prompt.lower():
            return '{"relevant": "yes", "confidence": 0.85}'
        elif "rewriter" in prompt.lower():
            return "What is the maximum number of HARQ processes in 5G NR Release 18?"
        elif "hallucination" in prompt.lower():
            return '{"grounded": "yes", "ungrounded_claims": []}'
        else:
            return "The maximum number of HARQ processes in NR is 16. [TS 38.321 §5.4]"
    return _mock_response


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for FAISS index."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_markdown_files) -> Path:
    """Provide temporary directory with sample markdown files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    for filename, content in sample_markdown_files.items():
        (data_dir / filename).write_text(content)
    
    return data_dir


@pytest.fixture
def sample_markdown_files():
    """Provide sample 3GPP markdown content."""
    return {
        "TS38.321.md": """# TS 38.321 - Medium Access Control (MAC)

## 5.4 HARQ Entity

### 5.4.1 HARQ Processes

The UE shall support a maximum of 16 HARQ processes per cell for FDD and TDD.

Each HARQ process handles one transport block at a time.
""",
        "TS38.331.md": """# TS 38.331 - Radio Resource Control (RRC)

## 5.3 RRC Connection Control

### 5.3.7 RRC Connection Re-establishment

The RRC connection re-establishment procedure is used to re-establish RRC 
connection after radio link failure.

Timer T311 is started upon detection of radio link failure.
""",
    }


# =============================================================================
# Graph State Fixtures
# =============================================================================

@pytest.fixture
def initial_graph_state(sample_question):
    """Provide initial graph state for testing."""
    from specagent.graph.state import create_initial_state
    return create_initial_state(sample_question)


@pytest.fixture
def state_after_retrieval(initial_graph_state, sample_chunks):
    """Provide graph state after retrieval step."""
    from specagent.graph.state import RetrievedChunk

    state = initial_graph_state.copy()
    state["route_decision"] = "retrieve"
    state["retrieved_chunks"] = [
        RetrievedChunk(
            content=chunk.content,
            spec_id=chunk.metadata.get("source_file", "").replace(".md", "").replace("-", "."),
            section=chunk.metadata.get("section_header", ""),
            similarity_score=0.85 - i * 0.1,
            chunk_id=f"{chunk.metadata.get('source_file', 'unknown')}:{chunk.metadata.get('chunk_index', i)}",
            source_file=chunk.metadata.get("source_file", ""),
        )
        for i, chunk in enumerate(sample_chunks[:3])
    ]
    return state


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def sample_benchmark_questions():
    """Provide sample benchmark questions for testing."""
    return [
        {
            "id": "q1",
            "question": "What is the maximum number of HARQ processes in NR?",
            "answer": "16",
            "difficulty": "easy",
        },
        {
            "id": "q2",
            "question": "What is the maximum number of component carriers for CA?",
            "answer": "16",
            "difficulty": "easy",
        },
        {
            "id": "q3",
            "question": "What timer is started upon radio link failure detection?",
            "answer": "T311",
            "difficulty": "medium",
        },
    ]


@pytest.fixture
def benchmark_file(tmp_path: Path, sample_benchmark_questions) -> Path:
    """Create temporary benchmark file."""
    benchmark_path = tmp_path / "benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(sample_benchmark_questions, f)
    return benchmark_path
