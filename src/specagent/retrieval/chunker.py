"""
Document chunking with metadata preservation.

Splits large markdown documents into smaller chunks while preserving:
    - Section headers for context
    - Source file information
    - Chunk position/index
"""

from dataclasses import dataclass, field

from specagent.config import settings


@dataclass
class Chunk:
    """A document chunk with metadata."""

    content: str
    """The text content of the chunk."""

    source_file: str
    """Original filename from TSpec-LLM dataset."""

    section_header: str = ""
    """The nearest section header above this chunk."""

    chunk_index: int = 0
    """Position of this chunk within the source document."""

    spec_id: str = ""
    """Extracted specification ID (e.g., 'TS38.331')."""

    section: str = ""
    """Extracted section number (e.g., '5.3.3')."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata."""

    @property
    def chunk_id(self) -> str:
        """Generate unique chunk identifier."""
        return f"{self.source_file}:{self.chunk_index}"


def chunk_markdown(
    text: str,
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split markdown text into chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter under the hood
    with markdown-aware separators.

    Args:
        text: Markdown document text
        source_file: Original filename for metadata
        chunk_size: Target chunk size in characters (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)

    Returns:
        List of Chunk objects with content and metadata
    """
    # TODO: Implement chunking logic
    # 1. Use RecursiveCharacterTextSplitter with markdown separators
    # 2. Extract section headers from markdown
    # 3. Parse spec_id and section from content/headers
    # 4. Create Chunk objects with metadata
    raise NotImplementedError("Chunker not yet implemented")


def extract_spec_id(text: str) -> str:
    """
    Extract 3GPP specification ID from text.

    Examples:
        "TS 38.331" -> "TS38.331"
        "3GPP TS 23.501" -> "TS23.501"

    Args:
        text: Text possibly containing spec reference

    Returns:
        Normalized spec ID or empty string
    """
    # TODO: Implement regex extraction
    raise NotImplementedError("Spec ID extraction not yet implemented")


def extract_section_header(text: str) -> str:
    """
    Extract the nearest markdown section header.

    Args:
        text: Markdown text

    Returns:
        Section header text or empty string
    """
    # TODO: Implement markdown header extraction
    raise NotImplementedError("Section header extraction not yet implemented")
