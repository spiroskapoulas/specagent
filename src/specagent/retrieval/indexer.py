"""
FAISS index management for vector similarity search.

Provides efficient similarity search over document chunk embeddings
with metadata storage and persistence.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from specagent.config import settings
from specagent.retrieval.chunker import Chunk


class FAISSIndex:
    """
    FAISS-based vector index for document retrieval.

    Uses IndexFlatIP (inner product) for cosine similarity search.
    Stores chunk metadata alongside vectors for result enrichment.

    Example:
        >>> index = FAISSIndex()
        >>> index.build(chunks, embeddings)
        >>> results = index.search(query_embedding, k=10)
        >>> index.save("data/index/faiss.index")
    """

    def __init__(self, dimension: int | None = None) -> None:
        """
        Initialize the FAISS index.

        Args:
            dimension: Vector dimension (default from settings)
        """
        self.dimension = dimension or settings.embedding_dimension
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self._index is not None

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def build(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        """
        Build the index from chunks and their embeddings.

        Args:
            chunks: List of document chunks
            embeddings: Array of shape (len(chunks), dimension)

        Raises:
            ValueError: If chunks and embeddings have different lengths
            ValueError: If embeddings have wrong dimension
        """
        # Validate inputs
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length: "
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings"
            )

        if len(embeddings) > 0 and embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must have dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        # Normalize embeddings for cosine similarity
        if len(embeddings) > 0:
            normalized_embeddings = self._normalize_embeddings(embeddings)
        else:
            normalized_embeddings = embeddings

        # Create FAISS IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self._index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to index
        if len(normalized_embeddings) > 0:
            # Ensure array is contiguous
            if not normalized_embeddings.flags['C_CONTIGUOUS']:
                normalized_embeddings = np.ascontiguousarray(normalized_embeddings)
            self._index.add(normalized_embeddings)

        # Store chunks for metadata lookup
        self._chunks = list(chunks)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        threshold: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            threshold: Minimum similarity score (optional)

        Returns:
            List of (chunk, similarity_score) tuples, sorted by score descending

        Raises:
            RuntimeError: If index has not been built
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call build() first.")

        # Type narrowing: assert index is not None after is_built check
        assert self._index is not None

        # Handle empty index
        if self.size == 0:
            return []

        # Normalize query embedding
        normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1))

        # Ensure array is contiguous
        if not normalized_query.flags['C_CONTIGUOUS']:
            normalized_query = np.ascontiguousarray(normalized_query)

        # Limit k to index size
        k = min(k, self.size)

        # Search FAISS index
        scores, indices = self._index.search(normalized_query, k)

        # Convert to list of (chunk, score) tuples
        results: list[tuple[Chunk, float]] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = float(scores[0][i])

            # Filter by threshold if provided
            if threshold is not None and score < threshold:
                continue

            chunk = self._chunks[idx]
            results.append((chunk, score))

        return results

    def save(self, path: str | Path | None = None) -> None:
        """
        Save index and metadata to disk.

        Args:
            path: Base path for index files (default from settings)

        Raises:
            RuntimeError: If index has not been built
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call build() first.")

        # Type narrowing: assert index is not None after is_built check
        assert self._index is not None

        # Use default path if not provided
        if path is None:
            path = settings.faiss_index_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = path.with_suffix('.index')
        faiss.write_index(self._index, str(index_file))

        # Save chunk metadata to JSON
        metadata_file = path.with_suffix('.json')
        chunks_data = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in self._chunks
        ]

        with metadata_file.open('w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path | None = None) -> None:
        """
        Load index and metadata from disk.

        Args:
            path: Base path to index files (default from settings)

        Raises:
            FileNotFoundError: If index files don't exist
        """
        # Use default path if not provided
        if path is None:
            path = settings.faiss_index_path

        path = Path(path)

        # Load FAISS index
        index_file = path.with_suffix('.index')
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        self._index = faiss.read_index(str(index_file))

        # Load chunk metadata from JSON
        metadata_file = path.with_suffix('.json')
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with metadata_file.open(encoding='utf-8') as f:
            chunks_data = json.load(f)

        self._chunks = [
            Chunk(
                content=chunk_data["content"],
                metadata=chunk_data["metadata"],
            )
            for chunk_data in chunks_data
        ]

    @classmethod
    def from_disk(cls, path: str | Path | None = None) -> "FAISSIndex":
        """
        Create index instance from saved files.

        Args:
            path: Path to index file

        Returns:
            FAISSIndex with loaded data
        """
        index = cls()
        index.load(path)
        return index

    def _normalize_embeddings(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize embeddings to unit length for cosine similarity.

        Args:
            embeddings: Array of shape (n, dimension)

        Returns:
            Normalized embeddings of same shape
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)
