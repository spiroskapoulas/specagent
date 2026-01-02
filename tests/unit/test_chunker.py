"""
Unit tests for document chunker module.

These tests verify:
    - Basic chunking behavior
    - Overlap handling
    - Metadata extraction
    - Edge cases (empty input, very long documents)
"""

import pytest

from specagent.retrieval.chunker import Chunk, chunk_markdown


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_id_generation(self):
        """chunk_id should be generated from source_file and index."""
        chunk = Chunk(
            content="Test content",
            source_file="TS38.321.md",
            chunk_index=5,
        )

        assert chunk.chunk_id == "TS38.321.md:5"

    def test_chunk_default_values(self):
        """Chunk should have sensible defaults."""
        chunk = Chunk(
            content="Test",
            source_file="test.md",
        )

        assert chunk.section_header == ""
        assert chunk.chunk_index == 0
        assert chunk.spec_id == ""
        assert chunk.section == ""
        assert chunk.metadata == {}


@pytest.mark.skip(reason="Chunker not yet implemented")
class TestChunkMarkdown:
    """Tests for chunk_markdown function."""

    def test_basic_chunking(self, sample_markdown_files):
        """Should split markdown into chunks."""
        content = sample_markdown_files["TS38.321.md"]

        chunks = chunk_markdown(
            text=content,
            source_file="TS38.321.md",
            chunk_size=200,
            chunk_overlap=50,
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self, sample_markdown_files):
        """Chunks should not exceed chunk_size."""
        content = sample_markdown_files["TS38.321.md"]
        chunk_size = 200

        chunks = chunk_markdown(
            text=content,
            source_file="TS38.321.md",
            chunk_size=chunk_size,
            chunk_overlap=0,
        )

        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 1.5  # Allow some flexibility

    def test_overlap_behavior(self, sample_markdown_files):
        """Consecutive chunks should have overlapping content."""
        content = sample_markdown_files["TS38.321.md"]

        chunks = chunk_markdown(
            text=content,
            source_file="TS38.321.md",
            chunk_size=200,
            chunk_overlap=50,
        )

        if len(chunks) >= 2:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                overlap = set(chunks[i].content[-50:]) & set(chunks[i + 1].content[:50])
                assert len(overlap) > 0

    def test_section_header_extraction(self, sample_markdown_files):
        """Should extract section headers into metadata."""
        content = sample_markdown_files["TS38.321.md"]

        chunks = chunk_markdown(
            text=content,
            source_file="TS38.321.md",
        )

        # At least some chunks should have section headers
        headers = [c.section_header for c in chunks if c.section_header]
        assert len(headers) > 0

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        chunks = chunk_markdown(
            text="",
            source_file="empty.md",
        )

        assert chunks == []

    def test_spec_id_extraction(self, sample_markdown_files):
        """Should extract spec ID from content/filename."""
        content = sample_markdown_files["TS38.321.md"]

        chunks = chunk_markdown(
            text=content,
            source_file="TS38.321.md",
        )

        # Should extract TS38.321 from filename
        for chunk in chunks:
            assert chunk.spec_id in ["TS38.321", ""]
