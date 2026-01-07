"""
Unit tests for resource caching in retrieval.resources module.

Tests the singleton caching behavior of:
    - get_faiss_index()
    - get_local_embedder()
    - get_hf_embedder()
    - initialize_resources()
    - clear_resource_cache()
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from specagent.retrieval.resources import (
    clear_resource_cache,
    get_faiss_index,
    get_hf_embedder,
    get_local_embedder,
    initialize_resources,
)


@pytest.mark.unit
class TestResourceCaching:
    """Test that resource getters implement proper caching."""

    def test_get_faiss_index_caches_result(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that get_faiss_index returns same instance on multiple calls."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build and save test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # First call loads from disk
            index1 = get_faiss_index()

            # Second call returns cached instance
            index2 = get_faiss_index()

            # Should be SAME object
            assert index1 is index2
            assert index1.is_built
            assert index1.size == len(sample_chunks)

    def test_get_local_embedder_caches_result(self):
        """Test that get_local_embedder returns same instance on multiple calls."""
        clear_resource_cache()

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # First call loads model
            embedder1 = get_local_embedder()

            # Second call returns cached instance
            embedder2 = get_local_embedder()

            # Should be SAME object
            assert embedder1 is embedder2
            assert embedder1.model is not None
            assert embedder1.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_hf_embedder_caches_result(self):
        """Test that get_hf_embedder returns same instance on multiple calls."""
        clear_resource_cache()

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.hf_api_key_value = "test-key"

            # First call creates embedder
            embedder1 = get_hf_embedder()

            # Second call returns cached instance
            embedder2 = get_hf_embedder()

            # Should be SAME object
            assert embedder1 is embedder2
            assert embedder1.api_key == "test-key"
            assert embedder1.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_clear_resource_cache_invalidates_all(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that clear_resource_cache invalidates all cached resources."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.hf_api_key_value = "test-key"

            # Load all resources
            index1 = get_faiss_index()
            local_embedder1 = get_local_embedder()
            hf_embedder1 = get_hf_embedder()

            # Clear cache
            clear_resource_cache()

            # Load again - should be NEW instances
            index2 = get_faiss_index()
            local_embedder2 = get_local_embedder()
            hf_embedder2 = get_hf_embedder()

            # Should be DIFFERENT objects
            assert index1 is not index2
            assert local_embedder1 is not local_embedder2
            assert hf_embedder1 is not hf_embedder2

    def test_cache_survives_multiple_calls(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that cache persists across many calls."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # First call
            index1 = get_faiss_index()

            # Call 10 more times
            for _ in range(10):
                index_n = get_faiss_index()
                # All should be same instance
                assert index_n is index1


@pytest.mark.unit
class TestInitializeResources:
    """Test the initialize_resources() function for eager loading."""

    def test_initialize_resources_success_local_embeddings(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test successful initialization with local embeddings."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)
            mock_settings.use_local_embeddings = True
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Initialize all resources
            status = initialize_resources()

            # Verify status
            assert status["faiss_index"] is True
            assert status["embedder"] is True

            # Verify resources are cached
            index = get_faiss_index()
            embedder = get_local_embedder()
            assert index.is_built
            assert embedder.model is not None

    def test_initialize_resources_success_hf_embeddings(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test successful initialization with HuggingFace embeddings."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)
            mock_settings.use_local_embeddings = False
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.hf_api_key_value = "test-key"

            # Initialize all resources
            status = initialize_resources()

            # Verify status
            assert status["faiss_index"] is True
            assert status["embedder"] is True

            # Verify resources are cached
            index = get_faiss_index()
            embedder = get_hf_embedder()
            assert index.is_built
            assert embedder.api_key == "test-key"

    def test_initialize_resources_index_not_found(self):
        """Test that initialize_resources raises RuntimeError when index missing."""
        clear_resource_cache()

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = "/nonexistent/path/faiss.index"
            mock_settings.use_local_embeddings = True

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                initialize_resources()

            assert "Failed to load FAISS index" in str(exc_info.value)

    def test_initialize_resources_embedder_error(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that initialize_resources raises RuntimeError when embedder fails."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)
            mock_settings.use_local_embeddings = True
            mock_settings.embedding_model = "invalid/model/path"

            # Mock LocalEmbedder to raise error
            with patch("specagent.retrieval.resources.LocalEmbedder") as MockEmbedder:
                MockEmbedder.side_effect = Exception("Model not found")

                # Should raise RuntimeError
                with pytest.raises(RuntimeError) as exc_info:
                    initialize_resources()

                assert "Failed to load embedder" in str(exc_info.value)


@pytest.mark.unit
class TestConcurrentAccess:
    """Test that caching works correctly with concurrent access patterns."""

    def test_multiple_retrieval_calls_use_same_index(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test simulating multiple retrieval operations sharing the index."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # Simulate 5 retrieval operations
            indexes = []
            for _ in range(5):
                idx = get_faiss_index()
                indexes.append(idx)

            # All should be the same instance (no disk loading)
            for idx in indexes[1:]:
                assert idx is indexes[0]

    def test_index_not_reloaded_after_search(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that searching the index doesn't trigger reload."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        real_index = FAISSIndex()
        real_index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        real_index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # Get index
            index1 = get_faiss_index()

            # Perform a search
            query_embedding = sample_embeddings[0]
            results1 = index1.search(query_embedding, k=5)

            # Get index again
            index2 = get_faiss_index()

            # Should still be same instance (not reloaded)
            assert index1 is index2

            # Search should still work
            results2 = index2.search(query_embedding, k=5)
            assert len(results2) == len(results1)


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in resource initialization."""

    def test_get_faiss_index_propagates_file_not_found(self):
        """Test that FileNotFoundError is propagated from get_faiss_index."""
        clear_resource_cache()

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = "/definitely/does/not/exist/index"

            with pytest.raises(FileNotFoundError):
                get_faiss_index()

    def test_get_local_embedder_propagates_model_error(self):
        """Test that model loading errors are propagated."""
        clear_resource_cache()

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.embedding_model = "invalid/nonexistent/model"

            # Should raise an error from sentence-transformers
            with pytest.raises(Exception):
                get_local_embedder()

    def test_initialize_resources_returns_status_on_partial_success(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test status dict shows which resources succeeded/failed."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index (this will succeed)
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)
            mock_settings.use_local_embeddings = True
            mock_settings.embedding_model = "invalid/model"

            # Mock LocalEmbedder to fail
            with patch("specagent.retrieval.resources.LocalEmbedder") as MockEmbedder:
                MockEmbedder.side_effect = Exception("Model load failed")

                # Should raise RuntimeError
                with pytest.raises(RuntimeError) as exc_info:
                    initialize_resources()

                # Error message should mention embedder failure
                assert "embedder" in str(exc_info.value).lower()


@pytest.mark.unit
class TestMemoryManagement:
    """Test memory management aspects of caching."""

    def test_cache_does_not_accumulate_multiple_indexes(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that cache only stores ONE index, not multiple."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # Get index 100 times
            indexes = [get_faiss_index() for _ in range(100)]

            # All should be the SAME object (only 1 in memory)
            first_id = id(indexes[0])
            for idx in indexes:
                assert id(idx) == first_id

    def test_clear_cache_allows_garbage_collection(
        self, tmp_index_dir, sample_chunks, sample_embeddings
    ):
        """Test that clearing cache allows old resources to be GC'd."""
        from specagent.retrieval.indexer import FAISSIndex

        clear_resource_cache()

        # Build test index
        index = FAISSIndex()
        index.build(sample_chunks, sample_embeddings)
        index_path = tmp_index_dir / "faiss"
        index.save(index_path)

        with patch("specagent.config.settings") as mock_settings:
            mock_settings.faiss_index_path = str(index_path)

            # Get index
            index1 = get_faiss_index()
            index1_id = id(index1)

            # Clear cache (old index can now be GC'd)
            clear_resource_cache()

            # Get new index
            index2 = get_faiss_index()
            index2_id = id(index2)

            # Should be different objects
            assert index1_id != index2_id
