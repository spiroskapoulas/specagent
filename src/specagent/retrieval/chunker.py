"""
Document chunking with metadata preservation.

Splits large markdown documents into smaller chunks while preserving:
    - Section headers for context
    - Source file information
    - Chunk position/index
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A document chunk with metadata."""

    content: str
    """The text content of the chunk."""

    metadata: dict[str, str | int] = field(default_factory=dict)
    """Metadata containing source_file, section_header, and chunk_index."""


def chunk_markdown(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """
    Split markdown text into chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter under the hood
    with markdown-aware separators.

    Args:
        text: Markdown document text to chunk
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of Chunk objects with content and metadata

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size
    """
    # Validate inputs
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    # Handle empty or whitespace-only input
    if not text or not text.strip():
        return []

    # Extract section headers from the entire document
    section_headers = _extract_section_headers(text)

    # Create text splitter with markdown-aware separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Double newline (paragraph breaks)
            "\n",  # Single newline
            " ",  # Space
            "",  # Character-level fallback
        ],
    )

    # Split the text
    text_chunks = splitter.split_text(text)

    # Create Chunk objects with metadata
    chunks: list[Chunk] = []
    for i, chunk_content in enumerate(text_chunks):
        # Find the nearest section header for this chunk
        section_header = _find_nearest_header(chunk_content, text, section_headers)

        chunk = Chunk(
            content=chunk_content,
            metadata={
                "source_file": "unknown",  # Default value
                "section_header": section_header,
                "chunk_index": i,
            },
        )
        chunks.append(chunk)

    return chunks


def _extract_section_headers(text: str) -> dict[int, str]:
    """
    Extract markdown section headers from text.

    Args:
        text: Markdown text to parse

    Returns:
        Dictionary mapping character position to header text
    """
    headers: dict[int, str] = {}
    # Match markdown headers (# Header, ## Header, etc.)
    pattern = r"^(#{1,6})\s+(.+)$"

    for match in re.finditer(pattern, text, re.MULTILINE):
        position = match.start()
        header_text = match.group(2).strip()
        headers[position] = header_text

    return headers


def _find_nearest_header(
    chunk_content: str, full_text: str, section_headers: dict[int, str]
) -> str:
    """
    Find the nearest section header for a chunk.

    Args:
        chunk_content: The content of the current chunk
        full_text: The full document text
        section_headers: Dictionary of header positions and text

    Returns:
        The nearest section header text, or empty string if none found
    """
    # First check if the chunk itself contains a header
    chunk_headers = _extract_section_headers(chunk_content)
    if chunk_headers:
        # Return the first header in the chunk
        first_position = min(chunk_headers.keys())
        return chunk_headers[first_position]

    # Find the chunk's position in the full text
    try:
        chunk_position = full_text.index(chunk_content)
    except ValueError:
        # Chunk not found in full text (shouldn't happen, but handle gracefully)
        return ""

    # Find the nearest header before this chunk
    nearest_header = ""
    nearest_position = -1

    for position, header_text in section_headers.items():
        if position < chunk_position and position > nearest_position:
            nearest_position = position
            nearest_header = header_text

    return nearest_header
