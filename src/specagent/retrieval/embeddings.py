"""
Embedding generation via HuggingFace Inference API.

Uses sentence-transformers models to generate vector embeddings
for document chunks and queries.
"""

import asyncio
import time
from typing import Optional

import httpx
import numpy as np
from numpy.typing import NDArray

from specagent.config import settings


class HuggingFaceEmbedder:
    """
    Generate embeddings using HuggingFace Inference API.

    Uses the free-tier API with retry logic for rate limits.

    Example:
        >>> embedder = HuggingFaceEmbedder()
        >>> vectors = embedder.embed_texts(["What is 5G NR?"])
        >>> vectors.shape
        (1, 384)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the embedder.

        Args:
            model: HuggingFace model ID (default from settings)
            api_key: HuggingFace API key (default from settings)
            batch_size: Number of texts per API call
        """
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.hf_api_key_value
        self.batch_size = batch_size
        self.dimension = settings.embedding_dimension
        self.base_url = "https://api-inference.huggingface.co/pipeline/feature-extraction"
        self.max_retries = 3
        self.initial_retry_delay = 1.0  # seconds

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Generate embeddings for a list of texts.

        Handles batching and retry logic for rate limits.

        Args:
            texts: List of texts to embed

        Returns:
            Array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings: list[NDArray[np.float32]] = []

        # Process texts in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch_sync(batch)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch results
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.dimension), dtype=np.float32)
        return result

    def _embed_batch_sync(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Embed a single batch of texts with retry logic.

        Args:
            texts: List of texts to embed (should be <= batch_size)

        Returns:
            Array of embeddings

        Raises:
            httpx.HTTPStatusError: If API returns non-429 error after retries
            httpx.HTTPError: If network error occurs
        """
        url = f"{self.base_url}/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": texts}

        retry_delay = self.initial_retry_delay

        with httpx.Client(timeout=30.0) as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.post(url, json=payload, headers=headers)

                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        if attempt < self.max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            response.raise_for_status()

                    # Raise for other HTTP errors
                    response.raise_for_status()

                    # Parse and normalize embeddings
                    embeddings = np.array(response.json(), dtype=np.float32)
                    return self._normalize_embeddings(embeddings)

                except httpx.HTTPStatusError as e:
                    # Only retry on 429 errors
                    if e.response.status_code != 429:
                        raise
                    # If this was the last attempt, raise
                    if attempt == self.max_retries - 1:
                        raise

        # This shouldn't be reached, but just in case
        raise RuntimeError("Unexpected error in embed_batch_sync")

    async def aembed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Async version of embed_texts for concurrent processing.

        Args:
            texts: List of texts to embed

        Returns:
            Array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Create batches
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        # Process all batches concurrently
        tasks = [self._embed_batch_async(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Concatenate all batch results
        result = np.vstack(batch_results) if batch_results else np.empty((0, self.dimension), dtype=np.float32)
        return result

    async def _embed_batch_async(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Async embed a single batch of texts with retry logic.

        Args:
            texts: List of texts to embed (should be <= batch_size)

        Returns:
            Array of embeddings

        Raises:
            httpx.HTTPStatusError: If API returns non-429 error after retries
            httpx.HTTPError: If network error occurs
        """
        url = f"{self.base_url}/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": texts}

        retry_delay = self.initial_retry_delay

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(url, json=payload, headers=headers)

                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            response.raise_for_status()

                    # Raise for other HTTP errors
                    response.raise_for_status()

                    # Parse and normalize embeddings
                    embeddings = np.array(response.json(), dtype=np.float32)
                    return self._normalize_embeddings(embeddings)

                except httpx.HTTPStatusError as e:
                    # Only retry on 429 errors
                    if e.response.status_code != 429:
                        raise
                    # If this was the last attempt, raise
                    if attempt == self.max_retries - 1:
                        raise

        # This shouldn't be reached, but just in case
        raise RuntimeError("Unexpected error in embed_batch_async")

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
        return embeddings / norms

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Array of shape (embedding_dimension,)
        """
        result = self.embed_texts([query])
        return result[0]
