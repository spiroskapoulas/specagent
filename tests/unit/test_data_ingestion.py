"""Unit tests for retrieval.data_ingestion module."""

from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from specagent.retrieval.data_ingestion import (
    download_all_required_specs,
    download_series_specs,
    download_specific_files,
)


@pytest.mark.unit
class TestDownloadSeriesSpecs:
    """Tests for downloading all files from a series."""

    def test_download_all_files_from_36_series(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test downloading all .md files from Rel-17/36_series."""
        output_dir = tmp_path / "raw_rel_17_36_series"

        # Mock the API listing response
        mock_file_list = [
            {"type": "file", "path": "3GPP-clean/Rel-17/36_series/36101-h60.md"},
            {"type": "file", "path": "3GPP-clean/Rel-17/36_series/36211-h60.md"},
            {"type": "file", "path": "3GPP-clean/Rel-17/36_series/36213-h60.md"},
            {"type": "directory", "path": "3GPP-clean/Rel-17/36_series/images"},
        ]

        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/36_series",
            json=mock_file_list,
        )

        # Mock file download responses
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/36_series/36101-h60.md",
            text="# TS 36.101 Content",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/36_series/36211-h60.md",
            text="# TS 36.211 Content",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/36_series/36213-h60.md",
            text="# TS 36.213 Content",
        )

        # Download files
        downloaded = download_series_specs(
            output_dir=output_dir,
            release="Rel-17",
            series="36_series",
            api_key="test-key",
        )

        # Verify results
        assert len(downloaded) == 3
        assert (output_dir / "36101-h60.md").exists()
        assert (output_dir / "36211-h60.md").exists()
        assert (output_dir / "36213-h60.md").exists()
        assert (output_dir / "36101-h60.md").read_text() == "# TS 36.101 Content"

    def test_download_all_files_from_38_series(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test downloading all .md files from Rel-17/38_series."""
        output_dir = tmp_path / "raw_rel_17_38_series"

        # Mock the API listing response
        mock_file_list = [
            {"type": "file", "path": "3GPP-clean/Rel-17/38_series/38201-h10.md"},
            {"type": "file", "path": "3GPP-clean/Rel-17/38_series/38211-h10.md"},
        ]

        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/38_series",
            json=mock_file_list,
        )

        # Mock file downloads
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/38_series/38201-h10.md",
            text="# TS 38.201 Content",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/38_series/38211-h10.md",
            text="# TS 38.211 Content",
        )

        downloaded = download_series_specs(
            output_dir=output_dir,
            release="Rel-17",
            series="38_series",
            api_key="test-key",
        )

        assert len(downloaded) == 2
        assert (output_dir / "38201-h10.md").exists()
        assert (output_dir / "38211-h10.md").exists()

    def test_download_creates_output_directory(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "nested" / "dir"

        # Mock empty file list
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/36_series",
            json=[],
        )

        download_series_specs(
            output_dir=output_dir,
            release="Rel-17",
            series="36_series",
            api_key="test-key",
        )

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_download_handles_api_error(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test handling of API errors."""
        output_dir = tmp_path / "output"

        # Mock API error
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/36_series",
            status_code=404,
        )

        with pytest.raises(httpx.HTTPStatusError):
            download_series_specs(
                output_dir=output_dir,
                release="Rel-17",
                series="36_series",
                api_key="test-key",
            )


@pytest.mark.unit
class TestDownloadSpecificFiles:
    """Tests for downloading specific files by name."""

    def test_download_specific_files_from_rel15_36_series(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ):
        """Test downloading specific files from Rel-15/36_series."""
        output_dir = tmp_path / "raw_rel_15_36_series"
        filenames = ["36777-f00_1.md", "36777-f00_2.md"]

        # Mock file downloads
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_1.md",
            text="# TS 36.777 Part 1",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_2.md",
            text="# TS 36.777 Part 2",
        )

        downloaded = download_specific_files(
            output_dir=output_dir,
            release="Rel-15",
            series="36_series",
            filenames=filenames,
            api_key="test-key",
        )

        assert len(downloaded) == 2
        assert (output_dir / "36777-f00_1.md").exists()
        assert (output_dir / "36777-f00_2.md").exists()
        assert (output_dir / "36777-f00_1.md").read_text() == "# TS 36.777 Part 1"

    def test_download_single_file(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test downloading a single specific file."""
        output_dir = tmp_path / "raw_rel_15_38_series"
        filenames = ["38811-f40.md"]

        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/38_series/38811-f40.md",
            text="# TS 38.811 Content",
        )

        downloaded = download_specific_files(
            output_dir=output_dir,
            release="Rel-15",
            series="38_series",
            filenames=filenames,
            api_key="test-key",
        )

        assert len(downloaded) == 1
        assert (output_dir / "38811-f40.md").exists()

    def test_download_handles_missing_file(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test handling of missing files (404)."""
        output_dir = tmp_path / "output"
        filenames = ["nonexistent.md"]

        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/nonexistent.md",
            status_code=404,
        )

        # Should continue despite error and return empty list
        downloaded = download_specific_files(
            output_dir=output_dir,
            release="Rel-15",
            series="36_series",
            filenames=filenames,
            api_key="test-key",
        )

        assert len(downloaded) == 0

    def test_download_multiple_files_from_rel16(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test downloading multiple specific files from Rel-16."""
        output_dir = tmp_path / "raw_rel_16_38_series"
        filenames = ["38821-g20.md", "38901-g10.md"]

        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38821-g20.md",
            text="# TS 38.821 Content",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38901-g10.md",
            text="# TS 38.901 Content",
        )

        downloaded = download_specific_files(
            output_dir=output_dir,
            release="Rel-16",
            series="38_series",
            filenames=filenames,
            api_key="test-key",
        )

        assert len(downloaded) == 2
        assert (output_dir / "38821-g20.md").exists()
        assert (output_dir / "38901-g10.md").exists()


@pytest.mark.unit
class TestDownloadAllRequiredSpecs:
    """Tests for the orchestration function that downloads all required specifications."""

    def test_download_all_required_specs(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test downloading all required specifications as per user requirements."""
        base_dir = tmp_path / "data"

        # Mock all API responses for Rel-17/36_series (all files)
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/36_series",
            json=[
                {"type": "file", "path": "3GPP-clean/Rel-17/36_series/36101-h60.md"},
            ],
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/36_series/36101-h60.md",
            text="# TS 36.101",
        )

        # Mock all API responses for Rel-17/38_series (all files)
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/38_series",
            json=[
                {"type": "file", "path": "3GPP-clean/Rel-17/38_series/38201-h10.md"},
            ],
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-17/38_series/38201-h10.md",
            text="# TS 38.201",
        )

        # Mock Rel-15/36_series specific files
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_1.md",
            text="# TS 36.777-1",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_2.md",
            text="# TS 36.777-2",
        )

        # Mock Rel-15/38_series specific file
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/38_series/38811-f40.md",
            text="# TS 38.811",
        )

        # Mock Rel-16/38_series specific files
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38821-g20.md",
            text="# TS 38.821",
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38901-g10.md",
            text="# TS 38.901",
        )

        # Run the orchestration function
        results = download_all_required_specs(base_dir=base_dir, api_key="test-key")

        # Verify all directories were created
        assert (base_dir / "raw_rel_17_36_series").exists()
        assert (base_dir / "raw_rel_17_38_series").exists()
        assert (base_dir / "raw_rel_15_36_series").exists()
        assert (base_dir / "raw_rel_15_38_series").exists()
        assert (base_dir / "raw_rel_16_38_series").exists()

        # Verify files were downloaded
        assert (base_dir / "raw_rel_17_36_series" / "36101-h60.md").exists()
        assert (base_dir / "raw_rel_17_38_series" / "38201-h10.md").exists()
        assert (base_dir / "raw_rel_15_36_series" / "36777-f00_1.md").exists()
        assert (base_dir / "raw_rel_15_36_series" / "36777-f00_2.md").exists()
        assert (base_dir / "raw_rel_15_38_series" / "38811-f40.md").exists()
        assert (base_dir / "raw_rel_16_38_series" / "38821-g20.md").exists()
        assert (base_dir / "raw_rel_16_38_series" / "38901-g10.md").exists()

        # Verify results structure
        assert "rel_17_36_series" in results
        assert "rel_17_38_series" in results
        assert "rel_15_36_series" in results
        assert "rel_15_38_series" in results
        assert "rel_16_38_series" in results

    def test_download_all_uses_settings_api_key(self, httpx_mock: HTTPXMock, tmp_path: Path):
        """Test that function uses API key from settings when not provided."""
        base_dir = tmp_path / "data"

        # Mock minimal responses
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/36_series",
            json=[],
        )
        httpx_mock.add_response(
            url="https://huggingface.co/api/datasets/rasoul-nikbakht/TSpec-LLM/tree/main/3GPP-clean/Rel-17/38_series",
            json=[],
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_1.md",
            status_code=404,
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/36_series/36777-f00_2.md",
            status_code=404,
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-15/38_series/38811-f40.md",
            status_code=404,
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38821-g20.md",
            status_code=404,
        )
        httpx_mock.add_response(
            url="https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM/resolve/main/3GPP-clean/Rel-16/38_series/38901-g10.md",
            status_code=404,
        )

        # Should not raise error even without api_key parameter
        results = download_all_required_specs(base_dir=base_dir)

        assert results is not None
