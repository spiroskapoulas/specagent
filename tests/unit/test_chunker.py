"""Unit tests for retrieval.chunker module."""

import pytest
from specagent.retrieval.chunker import (
    Chunk,
    chunk_markdown,
    _extract_section_headers,
    _find_nearest_header,
)


@pytest.mark.unit
class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk with all fields."""
        chunk = Chunk(
            content="Test content",
            metadata={
                "source_file": "test.md",
                "section_header": "Introduction",
                "chunk_index": 0,
            },
        )
        assert chunk.content == "Test content"
        assert chunk.metadata["source_file"] == "test.md"
        assert chunk.metadata["section_header"] == "Introduction"
        assert chunk.metadata["chunk_index"] == 0


@pytest.mark.unit
class TestChunkMarkdown:
    """Tests for chunk_markdown function."""

    def test_basic_chunking(self):
        """Test basic chunking of simple text."""
        text = "This is a simple text. " * 20  # ~440 chars
        chunks = chunk_markdown(text, chunk_size=200, overlap=0)

        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= 250 for chunk in chunks)  # Allow some flexibility

    def test_chunk_indices(self):
        """Test that chunks have correct indices."""
        text = "A" * 500
        chunks = chunk_markdown(text, chunk_size=100, overlap=0)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i

    def test_overlap_behavior(self):
        """Test that overlap creates overlapping content between chunks."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        chunks = chunk_markdown(text, chunk_size=100, overlap=20)

        # With overlap, consecutive chunks should share some content
        if len(chunks) > 1:
            # Check that we have more chunks with overlap than without
            chunks_no_overlap = chunk_markdown(text, chunk_size=100, overlap=0)
            assert len(chunks) >= len(chunks_no_overlap)

    def test_markdown_header_extraction(self):
        """Test extraction of markdown section headers."""
        text = """# Main Title

This is some content under the main title.

## Subsection 1

Content in subsection 1.

## Subsection 2

Content in subsection 2.
"""
        chunks = chunk_markdown(text, chunk_size=100, overlap=0)

        # At least one chunk should have captured a section header
        headers = [chunk.metadata.get("section_header") for chunk in chunks]
        assert any(header is not None for header in headers)

    def test_nested_markdown_headers(self):
        """Test extraction of nested markdown headers."""
        text = """# Level 1

Content at level 1.

## Level 2

Content at level 2.

### Level 3

Content at level 3.
"""
        chunks = chunk_markdown(text, chunk_size=50, overlap=0)

        # Should extract headers from different levels
        headers = [
            chunk.metadata.get("section_header")
            for chunk in chunks
            if chunk.metadata.get("section_header")
        ]
        assert len(headers) > 0

    def test_empty_input(self):
        """Test handling of empty input."""
        chunks = chunk_markdown("", chunk_size=100, overlap=0)
        assert chunks == []

    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        chunks = chunk_markdown("   \n\n\t  ", chunk_size=100, overlap=0)
        # Should return empty list or single chunk with empty/whitespace content
        assert len(chunks) <= 1
        if chunks:
            assert chunks[0].content.strip() == ""

    def test_text_smaller_than_chunk_size(self):
        """Test that small text returns single chunk."""
        text = "Short text"
        chunks = chunk_markdown(text, chunk_size=1000, overlap=0)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].metadata["chunk_index"] == 0

    def test_metadata_source_file_default(self):
        """Test that source_file metadata has default value."""
        text = "Some content"
        chunks = chunk_markdown(text, chunk_size=100, overlap=0)

        assert len(chunks) == 1
        assert "source_file" in chunks[0].metadata
        # Default should be "unknown" or similar
        assert chunks[0].metadata["source_file"] in ["unknown", ""]

    def test_chunk_size_validation(self):
        """Test that invalid chunk_size raises error."""
        text = "Some content"
        with pytest.raises((ValueError, AssertionError)):
            chunk_markdown(text, chunk_size=0, overlap=0)

    def test_overlap_validation(self):
        """Test that invalid overlap raises error."""
        text = "Some content"
        with pytest.raises((ValueError, AssertionError)):
            # Overlap larger than chunk_size should raise error
            chunk_markdown(text, chunk_size=100, overlap=150)

    def test_negative_overlap_validation(self):
        """Test that negative overlap raises ValueError."""
        text = "Some content"
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_markdown(text, chunk_size=100, overlap=-10)

    def test_special_characters_in_content(self):
        """Test handling of special characters."""
        text = """# Header with émojis 🚀

Content with special chars: <>&"'"""
        chunks = chunk_markdown(text, chunk_size=100, overlap=0)

        assert len(chunks) > 0
        # Content should be preserved
        assert "🚀" in "".join(chunk.content for chunk in chunks)

    def test_code_blocks_in_markdown(self):
        """Test handling of code blocks in markdown."""
        text = """# Code Example

```python
def hello():
    print("world")
```

More text after code.
"""
        chunks = chunk_markdown(text, chunk_size=100, overlap=0)

        assert len(chunks) > 0
        # Code block content should be preserved somewhere
        full_content = "".join(chunk.content for chunk in chunks)
        assert "def hello()" in full_content or "hello" in full_content


@pytest.mark.unit
class TestExtractSectionHeaders:
    """Tests for _extract_section_headers helper function."""

    def test_extract_multiple_headers(self):
        """Test extraction of multiple markdown headers."""
        text = """# Header 1

Content here.

## Header 2

More content.

### Header 3

Even more content.
"""
        headers = _extract_section_headers(text)

        assert len(headers) == 3
        assert "Header 1" in headers.values()
        assert "Header 2" in headers.values()
        assert "Header 3" in headers.values()

    def test_extract_no_headers(self):
        """Test extraction when no headers exist."""
        text = "Just plain text with no headers."
        headers = _extract_section_headers(text)

        assert len(headers) == 0
        assert headers == {}


@pytest.mark.unit
class TestFindNearestHeader:
    """Tests for _find_nearest_header helper function."""

    def test_chunk_not_found_in_full_text(self):
        """Test handling when chunk content is not found in full text."""
        chunk_content = "This chunk doesn't exist"
        full_text = "Completely different text"
        section_headers = {0: "Header"}

        result = _find_nearest_header(chunk_content, full_text, section_headers)

        # Should return empty string when chunk not found
        assert result == ""

    def test_no_headers_before_chunk(self):
        """Test when there are no headers before the chunk position."""
        # Create text where chunk appears before any headers
        full_text = """Plain text at the beginning without any headers.

# First Header

Content after header.
"""
        chunk_content = "Plain text at the beginning"
        # Extract headers - there's one header but it comes after our chunk
        section_headers = _extract_section_headers(full_text)

        result = _find_nearest_header(chunk_content, full_text, section_headers)

        # Should return empty string when no headers precede the chunk
        assert result == ""

    def test_chunk_contains_header(self):
        """Test when chunk itself contains a header."""
        chunk_content = """# My Header

Some content under the header.
"""
        full_text = chunk_content
        section_headers = _extract_section_headers(full_text)

        result = _find_nearest_header(chunk_content, full_text, section_headers)

        # Should return the header contained in the chunk
        assert result == "My Header"
