"""
Singleton resource management for FAISS index and embedders.

Provides cached instances of expensive resources that should only be
loaded once per application lifecycle. Uses @lru_cache pattern (same
as config.py settings singleton) to ensure resources are initialized
once and reused across all queries.

Key resources:
    - FAISS index (1.55GB, 30-60s load time)
    - LocalEmbedder (sentence-transformer model, ~500MB, 5-10s load time)
    - HuggingFaceEmbedder (API client, instant)

Usage:
    # In nodes or API handlers
    index = get_faiss_index()  # First call loads, subsequent calls instant
    embedder = get_local_embedder()  # First call loads, subsequent calls instant

    # In API startup (explicit initialization)
    status = initialize_resources()  # Preload all resources

    # In tests (reset cache)
    clear_resource_cache()  # Invalidate all cached resources
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from specagent.config import settings

if TYPE_CHECKING:
    from specagent.retrieval.embeddings import HuggingFaceEmbedder, LocalEmbedder
    from specagent.retrieval.indexer import FAISSIndex

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_faiss_index() -> "FAISSIndex":
    """
    Get or create the global FAISS index instance.

    First call loads the index from disk (30-60s, 1.55GB).
    Subsequent calls return the cached instance (instant).

    Returns:
        FAISSIndex: Loaded index ready for search

    Raises:
        FileNotFoundError: If index file doesn't exist at configured path
        RuntimeError: If index loading fails

    Example:
        >>> index = get_faiss_index()
        >>> results = index.search(query_embedding, k=10)
    """
    from specagent.retrieval.indexer import FAISSIndex

    logger.info(
        f"Loading FAISS index from {settings.faiss_index_path} "
        "(this may take 30-60 seconds)..."
    )

    index = FAISSIndex()
    index.load(settings.faiss_index_path)

    logger.info(f"FAISS index loaded successfully ({index.size} vectors)")

    return index


@lru_cache(maxsize=1)
def get_local_embedder() -> "LocalEmbedder":
    """
    Get or create the global LocalEmbedder instance.

    First call loads the sentence-transformer model (5-10s, ~500MB).
    Subsequent calls return the cached instance (instant).

    Returns:
        LocalEmbedder: Initialized embedder ready for use

    Example:
        >>> embedder = get_local_embedder()
        >>> embedding = embedder.embed_query("What is 5G NR?")
    """
    from specagent.retrieval.embeddings import LocalEmbedder

    logger.info(
        f"Loading local embedder model: {settings.embedding_model} "
        "(this may take 5-10 seconds)..."
    )

    embedder = LocalEmbedder(
        model=settings.embedding_model,
        batch_size=32,
        show_progress=False,  # No progress bar for cached instance
    )

    logger.info(f"Local embedder loaded successfully ({embedder.model_name})")

    return embedder


@lru_cache(maxsize=1)
def get_hf_embedder() -> "HuggingFaceEmbedder":
    """
    Get or create the global HuggingFaceEmbedder instance.

    Initializes the HuggingFace Inference API client (instant).
    Subsequent calls return the cached instance.

    Returns:
        HuggingFaceEmbedder: Initialized embedder ready for use

    Example:
        >>> embedder = get_hf_embedder()
        >>> embeddings = await embedder.aembed_texts(["What is 5G NR?"])
    """
    from specagent.retrieval.embeddings import HuggingFaceEmbedder

    logger.info(
        f"Initializing HuggingFace embedder for model: {settings.embedding_model}"
    )

    embedder = HuggingFaceEmbedder(
        model=settings.embedding_model,
        api_key=settings.hf_api_key_value,
        batch_size=32,
    )

    logger.info("HuggingFace embedder initialized successfully")

    return embedder


def initialize_resources() -> dict[str, bool]:
    """
    Explicitly initialize all resources for eager loading.

    Called at API server startup to front-load expensive operations
    before handling requests. For CLI, resources lazy-load instead.

    Returns:
        dict: Status of each resource initialization
            - "faiss_index": True if loaded successfully
            - "embedder": True if loaded successfully

    Raises:
        RuntimeError: If any resource fails to initialize

    Example:
        >>> # In FastAPI lifespan handler
        >>> status = initialize_resources()
        >>> logger.info(f"Resources loaded: {status}")
    """
    status = {}

    # Load FAISS index
    try:
        index = get_faiss_index()
        status["faiss_index"] = index.is_built
    except Exception as e:
        status["faiss_index"] = False
        raise RuntimeError(f"Failed to load FAISS index: {e}") from e

    # Load embedder based on config
    try:
        if settings.use_local_embeddings:
            embedder = get_local_embedder()
            status["embedder"] = embedder.model is not None
        else:
            embedder = get_hf_embedder()
            status["embedder"] = embedder.api_key is not None
    except Exception as e:
        status["embedder"] = False
        raise RuntimeError(f"Failed to load embedder: {e}") from e

    return status


def clear_resource_cache() -> None:
    """
    Clear all cached resources.

    Used in tests to reset state between test cases.
    In production, resources persist for application lifetime.

    Example:
        >>> # In pytest fixture
        >>> clear_resource_cache()
        >>> # Now get_faiss_index() will reload from disk
    """
    get_faiss_index.cache_clear()
    get_local_embedder.cache_clear()
    get_hf_embedder.cache_clear()
    logger.debug("Resource cache cleared")
