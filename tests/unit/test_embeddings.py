"""Unit tests for retrieval.embeddings module."""

import asyncio
import json
import time

import httpx
import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from specagent.retrieval.embeddings import HuggingFaceEmbedder


@pytest.mark.unit
class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder class."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        embedder = HuggingFaceEmbedder()

        # Just verify it initializes without error and has expected attributes
        assert embedder.model is not None
        assert embedder.batch_size == 32
        assert embedder.dimension > 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        embedder = HuggingFaceEmbedder(
            model="custom-model",
            api_key="custom-key",
            batch_size=16,
        )

        assert embedder.model == "custom-model"
        assert embedder.api_key == "custom-key"
        assert embedder.batch_size == 16

    def test_embed_texts_single_text(self, httpx_mock: HTTPXMock):
        """Test embedding a single text."""
        # Mock the HuggingFace API response
        mock_embedding = [[0.1] * 384]
        httpx_mock.add_response(
            url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            json=mock_embedding,
        )

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts(["test text"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)
        assert result.dtype == np.float32

    def test_embed_texts_multiple_texts(self, httpx_mock: HTTPXMock):
        """Test embedding multiple texts."""
        # Mock the HuggingFace API response
        mock_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        httpx_mock.add_response(
            url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            json=mock_embeddings,
        )

        embedder = HuggingFaceEmbedder(api_key="test-key")
        texts = ["text 1", "text 2", "text 3"]
        result = embedder.embed_texts(texts)

        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_embed_texts_batching(self, httpx_mock: HTTPXMock):
        """Test that texts are batched correctly."""
        # Create embedder with small batch size
        embedder = HuggingFaceEmbedder(api_key="test-key", batch_size=2)

        # Mock two batch requests
        mock_embedding_batch1 = [[0.1] * 384, [0.2] * 384]
        mock_embedding_batch2 = [[0.3] * 384]

        httpx_mock.add_response(json=mock_embedding_batch1)
        httpx_mock.add_response(json=mock_embedding_batch2)

        texts = ["text 1", "text 2", "text 3"]
        result = embedder.embed_texts(texts)

        # Should make 2 requests (batch of 2, then batch of 1)
        assert len(httpx_mock.get_requests()) == 2
        assert result.shape == (3, 384)

    def test_embed_texts_empty_list(self):
        """Test embedding an empty list."""
        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)

    def test_embed_texts_normalization(self, httpx_mock: HTTPXMock):
        """Test that embeddings are normalized."""
        # Mock unnormalized embeddings
        mock_embedding = [[3.0, 4.0] + [0.0] * 382]  # Length = 5 before normalization
        httpx_mock.add_response(json=mock_embedding)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts(["test"])

        # Check that vectors are normalized (L2 norm = 1)
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embed_texts_retry_on_429(self, httpx_mock: HTTPXMock):
        """Test retry logic on rate limit (429 error)."""
        # First request returns 429, second succeeds
        httpx_mock.add_response(status_code=429, json={"error": "Rate limit exceeded"})
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts(["test"])

        # Should have made 2 requests (1 failed, 1 succeeded)
        assert len(httpx_mock.get_requests()) == 2
        assert result.shape == (1, 384)

    def test_embed_texts_retry_exponential_backoff(self, httpx_mock: HTTPXMock, monkeypatch):
        """Test that retry uses exponential backoff."""
        sleep_times = []

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        monkeypatch.setattr(time, "sleep", mock_sleep)

        # Return 429 twice, then succeed
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts(["test"])

        # Should have slept with exponential backoff
        assert len(sleep_times) == 2
        # Second sleep should be longer than first (exponential backoff)
        assert sleep_times[1] > sleep_times[0]
        assert result.shape == (1, 384)

    def test_embed_texts_max_retries_exceeded(self, httpx_mock: HTTPXMock):
        """Test that max retries are respected."""
        # Return 429 for all retry attempts (max_retries = 3)
        for _ in range(3):
            httpx_mock.add_response(status_code=429)

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_texts(["test"])

    def test_embed_texts_http_error(self, httpx_mock: HTTPXMock):
        """Test handling of HTTP errors other than 429."""
        httpx_mock.add_response(status_code=500, json={"error": "Internal server error"})

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_texts(["test"])

    def test_embed_texts_network_error(self, httpx_mock: HTTPXMock):
        """Test handling of network errors."""
        httpx_mock.add_exception(httpx.ConnectError("Connection failed"))

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.ConnectError):
            embedder.embed_texts(["test"])

    def test_embed_query(self, httpx_mock: HTTPXMock):
        """Test embedding a single query."""
        mock_embedding = [[0.1] * 384]
        httpx_mock.add_response(json=mock_embedding)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_query("test query")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_texts_single_text(self, httpx_mock: HTTPXMock):
        """Test async embedding a single text."""
        mock_embedding = [[0.1] * 384]
        httpx_mock.add_response(json=mock_embedding)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = await embedder.aembed_texts(["test text"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_texts_multiple_texts(self, httpx_mock: HTTPXMock):
        """Test async embedding multiple texts."""
        mock_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        httpx_mock.add_response(json=mock_embeddings)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        texts = ["text 1", "text 2", "text 3"]
        result = await embedder.aembed_texts(texts)

        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_aembed_texts_batching(self, httpx_mock: HTTPXMock):
        """Test async batching of texts."""
        embedder = HuggingFaceEmbedder(api_key="test-key", batch_size=2)

        # Mock two batch requests
        httpx_mock.add_response(json=[[0.1] * 384, [0.2] * 384])
        httpx_mock.add_response(json=[[0.3] * 384])

        texts = ["text 1", "text 2", "text 3"]
        result = await embedder.aembed_texts(texts)

        assert len(httpx_mock.get_requests()) == 2
        assert result.shape == (3, 384)

    @pytest.mark.asyncio
    async def test_aembed_texts_retry_on_429(self, httpx_mock: HTTPXMock):
        """Test async retry logic on rate limit."""
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = await embedder.aembed_texts(["test"])

        assert len(httpx_mock.get_requests()) == 2
        assert result.shape == (1, 384)

    @pytest.mark.asyncio
    async def test_aembed_texts_concurrent_batches(self, httpx_mock: HTTPXMock):
        """Test that async version processes batches concurrently."""
        embedder = HuggingFaceEmbedder(api_key="test-key", batch_size=2)

        # Mock three batch requests
        for _ in range(3):
            httpx_mock.add_response(json=[[0.1] * 384, [0.2] * 384])

        texts = ["text"] * 6
        result = await embedder.aembed_texts(texts)

        # Should process all batches
        assert result.shape == (6, 384)

    def test_api_request_headers(self, httpx_mock: HTTPXMock):
        """Test that API requests include correct headers."""
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-api-key")
        embedder.embed_texts(["test"])

        request = httpx_mock.get_requests()[0]
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert "application/json" in request.headers["Content-Type"]

    def test_api_request_body(self, httpx_mock: HTTPXMock):
        """Test that API request body is formatted correctly."""
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-key")
        embedder.embed_texts(["test text"])

        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert body["inputs"] == ["test text"]

    def test_custom_model_url(self, httpx_mock: HTTPXMock):
        """Test that custom model uses correct API URL."""
        custom_model = "custom-org/custom-model"
        httpx_mock.add_response(
            url=f"https://api-inference.huggingface.co/pipeline/feature-extraction/{custom_model}",
            json=[[0.1] * 384],
        )

        embedder = HuggingFaceEmbedder(model=custom_model, api_key="test-key")
        embedder.embed_texts(["test"])

        request = httpx_mock.get_requests()[0]
        assert custom_model in str(request.url)

    @pytest.mark.asyncio
    async def test_aembed_texts_empty_list(self):
        """Test async embedding an empty list."""
        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = await embedder.aembed_texts([])

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)

    @pytest.mark.asyncio
    async def test_aembed_texts_max_retries_exceeded(self, httpx_mock: HTTPXMock):
        """Test async max retries handling."""
        # Return 429 for all retry attempts (max_retries = 3)
        for _ in range(3):
            httpx_mock.add_response(status_code=429)

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.HTTPStatusError):
            await embedder.aembed_texts(["test"])

    @pytest.mark.asyncio
    async def test_aembed_texts_http_error(self, httpx_mock: HTTPXMock):
        """Test async handling of HTTP errors other than 429."""
        httpx_mock.add_response(status_code=500, json={"error": "Internal server error"})

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.HTTPStatusError):
            await embedder.aembed_texts(["test"])

    @pytest.mark.asyncio
    async def test_aembed_texts_network_error(self, httpx_mock: HTTPXMock):
        """Test async handling of network errors."""
        httpx_mock.add_exception(httpx.ConnectError("Connection failed"))

        embedder = HuggingFaceEmbedder(api_key="test-key")

        with pytest.raises(httpx.ConnectError):
            await embedder.aembed_texts(["test"])

    @pytest.mark.asyncio
    async def test_aembed_texts_exponential_backoff(self, httpx_mock: HTTPXMock, monkeypatch):
        """Test async retry uses exponential backoff."""
        sleep_times = []

        async def mock_sleep(seconds):
            sleep_times.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        # Return 429 twice, then succeed
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=[[0.1] * 384])

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = await embedder.aembed_texts(["test"])

        # Should have slept with exponential backoff
        assert len(sleep_times) == 2
        # Second sleep should be longer than first (exponential backoff)
        assert sleep_times[1] > sleep_times[0]
        assert result.shape == (1, 384)

    def test_normalize_zero_vector(self, httpx_mock: HTTPXMock):
        """Test normalization handles zero vectors."""
        mock_embedding = [[0.0] * 384]  # All zeros
        httpx_mock.add_response(json=mock_embedding)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = embedder.embed_texts(["test"])

        # Should not raise, should return zero vector unchanged
        assert result.shape == (1, 384)
        assert np.allclose(result[0], 0.0)

    @pytest.mark.asyncio
    async def test_aembed_texts_normalization(self, httpx_mock: HTTPXMock):
        """Test async embeddings are normalized."""
        # Mock unnormalized embeddings
        mock_embedding = [[3.0, 4.0] + [0.0] * 382]  # Length = 5 before normalization
        httpx_mock.add_response(json=mock_embedding)

        embedder = HuggingFaceEmbedder(api_key="test-key")
        result = await embedder.aembed_texts(["test"])

        # Check that vectors are normalized (L2 norm = 1)
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, atol=1e-5)
