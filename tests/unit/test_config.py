"""
Unit tests for configuration module.
"""

import os
from unittest.mock import patch

import pytest


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_from_env(self, mock_settings):
        """Settings should load values from environment variables."""
        assert mock_settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert mock_settings.chunk_size == 512
        assert mock_settings.chunk_overlap == 64
        assert mock_settings.enable_tracing is False

    def test_settings_validates_api_key(self):
        """Settings should require HF_API_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            from specagent.config import Settings
            
            with pytest.raises(Exception):  # ValidationError
                Settings()

    def test_settings_chunk_overlap_validation(self):
        """Chunk overlap must be less than chunk size."""
        with patch.dict(
            os.environ,
            {
                "HF_API_KEY": "test-key",
                "CHUNK_SIZE": "256",
                "CHUNK_OVERLAP": "300",  # Invalid: > chunk_size
            },
            clear=True,
        ):
            from specagent.config import Settings
            
            with pytest.raises(Exception):  # ValidationError
                Settings()

    def test_settings_hf_api_key_is_secret(self, mock_settings):
        """API key should be stored as SecretStr."""
        # Direct access should not reveal the value
        assert "test-api-key" not in str(mock_settings.hf_api_key)
        
        # Explicit method should reveal the value
        assert mock_settings.hf_api_key_value == "test-api-key"

    def test_settings_paths_are_resolved(self, mock_settings):
        """Path settings should be resolved to absolute paths."""
        assert mock_settings.faiss_index_path.is_absolute()
        assert mock_settings.data_dir.is_absolute()

    def test_get_settings_is_cached(self):
        """get_settings should return cached instance."""
        with patch.dict(os.environ, {"HF_API_KEY": "test-key"}):
            from specagent.config import get_settings

            # Clear cache first
            get_settings.cache_clear()

            settings1 = get_settings()
            settings2 = get_settings()

            assert settings1 is settings2
