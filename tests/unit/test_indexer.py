"""Unit tests for retrieval.indexer module."""

from unittest.mock import patch

import numpy as np
import pytest

from specagent.retrieval.chunker import Chunk
from specagent.retrieval.indexer import FAISSIndex


@pytest.mark.unit
class TestFAISSIndex:
    """Tests for FAISSIndex class."""

    def test_init_default_dimension(self):
        """Test initialization with default dimension."""
        index = FAISSIndex()

        assert index.dimension > 0
        assert not index.is_built
        assert index.size == 0

    def test_init_custom_dimension(self):
        """Test initialization with custom dimension."""
        index = FAISSIndex(dimension=512)

        assert index.dimension == 512
        assert not index.is_built
        assert index.size == 0

    def test_build_with_valid_inputs(self):
        """Test building index with valid chunks and embeddings."""
        chunks = [
            Chunk(content="test 1", metadata={"source_file": "test.md", "chunk_index": 0}),
            Chunk(content="test 2", metadata={"source_file": "test.md", "chunk_index": 1}),
            Chunk(content="test 3", metadata={"source_file": "test.md", "chunk_index": 2}),
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        assert index.is_built
        assert index.size == 3

    def test_build_normalizes_embeddings(self):
        """Test that build normalizes embeddings for cosine similarity."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        # Create unnormalized embedding
        embeddings = np.array([[3.0, 4.0] + [0.0] * 382], dtype=np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        # Search with same unnormalized vector
        query = embeddings[0]
        results = index.search(query, k=1)

        # Should find exact match with score close to 1.0
        assert len(results) == 1
        assert results[0][0].content == "test"
        assert results[0][1] > 0.99  # Normalized cosine similarity

    def test_build_with_mismatched_lengths(self):
        """Test that build raises error when chunks and embeddings have different lengths."""
        chunks = [
            Chunk(content="test 1", metadata={"source_file": "test.md", "chunk_index": 0}),
            Chunk(content="test 2", metadata={"source_file": "test.md", "chunk_index": 1}),
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)

        with pytest.raises(ValueError):
            index.build(chunks, embeddings)

    def test_build_with_wrong_dimension(self):
        """Test that build raises error when embeddings have wrong dimension."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 512).astype(np.float32)

        index = FAISSIndex(dimension=384)

        with pytest.raises(ValueError):
            index.build(chunks, embeddings)

    def test_build_empty_inputs(self):
        """Test building index with empty inputs."""
        index = FAISSIndex(dimension=384)
        index.build([], np.empty((0, 384), dtype=np.float32))

        assert index.is_built
        assert index.size == 0

    def test_search_returns_correct_results(self):
        """Test that search returns correct chunks."""
        chunks = [
            Chunk(content="machine learning", metadata={"source_file": "ml.md", "chunk_index": 0}),
            Chunk(content="deep learning", metadata={"source_file": "dl.md", "chunk_index": 0}),
            Chunk(content="cooking recipes", metadata={"source_file": "cooking.md", "chunk_index": 0}),
        ]
        # Create embeddings where first two are similar, third is different
        embeddings = np.array([
            [1.0, 0.0] + [0.0] * 382,
            [0.9, 0.1] + [0.0] * 382,
            [0.0, 1.0] + [0.0] * 382,
        ], dtype=np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        # Search with query similar to first embedding
        query = np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)
        results = index.search(query, k=2)

        assert len(results) == 2
        # First result should be "machine learning"
        assert results[0][0].content == "machine learning"
        # Second result should be "deep learning"
        assert results[1][0].content == "deep learning"

    def test_search_returns_scores_in_descending_order(self):
        """Test that search results are sorted by score descending."""
        chunks = [
            Chunk(content=f"chunk {i}", metadata={"source_file": "test.md", "chunk_index": i})
            for i in range(5)
        ]
        embeddings = np.random.rand(5, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = np.random.rand(384).astype(np.float32)
        results = index.search(query, k=5)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_k_parameter(self):
        """Test that search returns exactly k results."""
        chunks = [
            Chunk(content=f"chunk {i}", metadata={"source_file": "test.md", "chunk_index": i})
            for i in range(10)
        ]
        embeddings = np.random.rand(10, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = np.random.rand(384).astype(np.float32)

        results_3 = index.search(query, k=3)
        assert len(results_3) == 3

        results_7 = index.search(query, k=7)
        assert len(results_7) == 7

    def test_search_with_k_larger_than_index_size(self):
        """Test search when k is larger than index size."""
        chunks = [
            Chunk(content=f"chunk {i}", metadata={"source_file": "test.md", "chunk_index": i})
            for i in range(3)
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = np.random.rand(384).astype(np.float32)
        results = index.search(query, k=10)

        # Should return only 3 results (all chunks in index)
        assert len(results) == 3

    def test_search_with_threshold(self):
        """Test that search filters results by threshold."""
        chunks = [
            Chunk(content="similar", metadata={"source_file": "test.md", "chunk_index": 0}),
            Chunk(content="different", metadata={"source_file": "test.md", "chunk_index": 1}),
        ]
        embeddings = np.array([
            [1.0, 0.0] + [0.0] * 382,
            [0.0, 1.0] + [0.0] * 382,
        ], dtype=np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)
        results = index.search(query, k=10, threshold=0.5)

        # Only the similar chunk should pass threshold
        assert len(results) >= 1
        assert all(score >= 0.5 for _, score in results)

    def test_search_before_build_raises_error(self):
        """Test that search raises error if index not built."""
        index = FAISSIndex(dimension=384)
        query = np.random.rand(384).astype(np.float32)

        with pytest.raises(RuntimeError):
            index.search(query, k=5)

    def test_search_empty_index(self):
        """Test search on empty index."""
        index = FAISSIndex(dimension=384)
        index.build([], np.empty((0, 384), dtype=np.float32))

        query = np.random.rand(384).astype(np.float32)
        results = index.search(query, k=5)

        assert results == []

    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        chunks = [
            Chunk(content="test 1", metadata={"source_file": "test.md", "chunk_index": 0}),
            Chunk(content="test 2", metadata={"source_file": "test.md", "chunk_index": 1}),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)

        # Build and save
        index1 = FAISSIndex(dimension=384)
        index1.build(chunks, embeddings)
        save_path = tmp_path / "test_index"
        index1.save(save_path)

        # Load
        index2 = FAISSIndex(dimension=384)
        index2.load(save_path)

        assert index2.is_built
        assert index2.size == 2

        # Verify search works on loaded index
        query = embeddings[0]
        results = index2.search(query, k=1)
        assert len(results) == 1
        assert results[0][0].content == "test 1"

    def test_save_preserves_metadata(self, tmp_path):
        """Test that save preserves chunk metadata."""
        chunks = [
            Chunk(
                content="test content",
                metadata={
                    "source_file": "TS38.321.md",
                    "section_header": "5.4 HARQ",
                    "chunk_index": 42,
                }
            ),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        index1 = FAISSIndex(dimension=384)
        index1.build(chunks, embeddings)
        save_path = tmp_path / "metadata_test"
        index1.save(save_path)

        index2 = FAISSIndex(dimension=384)
        index2.load(save_path)

        query = embeddings[0]
        results = index2.search(query, k=1)

        assert results[0][0].content == "test content"
        assert results[0][0].metadata["source_file"] == "TS38.321.md"
        assert results[0][0].metadata["section_header"] == "5.4 HARQ"
        assert results[0][0].metadata["chunk_index"] == 42

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Test that load raises error for nonexistent file."""
        index = FAISSIndex(dimension=384)
        nonexistent_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            index.load(nonexistent_path)

    def test_save_before_build_raises_error(self, tmp_path):
        """Test that save raises error if index not built."""
        index = FAISSIndex(dimension=384)
        save_path = tmp_path / "test"

        with pytest.raises(RuntimeError):
            index.save(save_path)

    def test_from_disk_classmethod(self, tmp_path):
        """Test from_disk class method."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        index1 = FAISSIndex(dimension=384)
        index1.build(chunks, embeddings)
        save_path = tmp_path / "from_disk_test"
        index1.save(save_path)

        # Load using class method
        index2 = FAISSIndex.from_disk(save_path)

        assert index2.is_built
        assert index2.size == 1
        assert index2.dimension == 384

    def test_multiple_builds_replace_index(self):
        """Test that building multiple times replaces the index."""
        chunks1 = [
            Chunk(content="first", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings1 = np.random.rand(1, 384).astype(np.float32)

        chunks2 = [
            Chunk(content="second 1", metadata={"source_file": "test.md", "chunk_index": 0}),
            Chunk(content="second 2", metadata={"source_file": "test.md", "chunk_index": 1}),
        ]
        embeddings2 = np.random.rand(2, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)

        # First build
        index.build(chunks1, embeddings1)
        assert index.size == 1

        # Second build should replace
        index.build(chunks2, embeddings2)
        assert index.size == 2

    def test_search_returns_chunk_and_score_tuples(self):
        """Test that search returns tuples of (Chunk, float)."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = embeddings[0]
        results = index.search(query, k=1)

        assert len(results) == 1
        chunk, score = results[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_build_with_non_contiguous_array(self):
        """Test that build works with non-contiguous arrays."""
        chunks = [
            Chunk(content=f"chunk {i}", metadata={"source_file": "test.md", "chunk_index": i})
            for i in range(3)
        ]
        # Create non-contiguous array via transpose
        embeddings = np.random.rand(384, 3).astype(np.float32).T
        assert not embeddings.flags['C_CONTIGUOUS']

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        assert index.is_built
        assert index.size == 3

    def test_search_with_non_contiguous_query(self):
        """Test that search works with non-contiguous query arrays."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        query = np.random.rand(384).astype(np.float32)

        # Mock _normalize_embeddings to return a guaranteed non-contiguous array
        original_normalize = index._normalize_embeddings

        def mock_normalize(emb):
            # Call the original normalization
            result = original_normalize(emb)
            # Create a Fortran-ordered array (not C-contiguous)
            fortran_array = np.asfortranarray(result)
            # Verify it's not C-contiguous
            if not fortran_array.flags['C_CONTIGUOUS']:
                return fortran_array
            # If that didn't work, create from a larger array slice
            larger = np.zeros((3, result.shape[1]), dtype=np.float32, order='F')
            larger[1] = result[0]
            return larger[1:2, :]  # Slice of Fortran array is non-contiguous

        with patch.object(index, '_normalize_embeddings', side_effect=mock_normalize):
            results = index.search(query, k=1)

        assert len(results) == 1
        assert isinstance(results[0][0], Chunk)
        assert isinstance(results[0][1], float)

    def test_save_with_default_path(self, tmp_path, monkeypatch):
        """Test save with default path from settings."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        # Mock settings to use tmp_path
        from specagent import config
        monkeypatch.setattr(config.settings, 'faiss_index_path', tmp_path / "default_index")

        index = FAISSIndex(dimension=384)
        index.build(chunks, embeddings)

        # Call save without path argument
        index.save()

        # Verify files were created at default location
        assert (tmp_path / "default_index.index").exists()
        assert (tmp_path / "default_index.json").exists()

    def test_load_with_default_path(self, tmp_path, monkeypatch):
        """Test load with default path from settings."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        # Mock settings to use tmp_path
        from specagent import config
        save_path = tmp_path / "default_load_index"
        monkeypatch.setattr(config.settings, 'faiss_index_path', save_path)

        # Build and save
        index1 = FAISSIndex(dimension=384)
        index1.build(chunks, embeddings)
        index1.save(save_path)

        # Load using default path
        index2 = FAISSIndex(dimension=384)
        index2.load()

        assert index2.is_built
        assert index2.size == 1

    def test_load_missing_metadata_file(self, tmp_path):
        """Test that load raises error when metadata file is missing."""
        chunks = [
            Chunk(content="test", metadata={"source_file": "test.md", "chunk_index": 0}),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)

        # Save index
        index1 = FAISSIndex(dimension=384)
        index1.build(chunks, embeddings)
        save_path = tmp_path / "incomplete_index"
        index1.save(save_path)

        # Delete metadata file
        metadata_file = save_path.with_suffix('.json')
        metadata_file.unlink()

        # Try to load
        index2 = FAISSIndex(dimension=384)
        with pytest.raises(FileNotFoundError) as excinfo:
            index2.load(save_path)

        assert "Metadata file not found" in str(excinfo.value)
