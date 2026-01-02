# Implement Retrieval Component

Implement a retrieval pipeline component (chunker, embeddings, or indexer).

## Process

1. **Identify component**: chunker | embeddings | indexer
2. **Review placeholder** in `src/specagent/retrieval/{component}.py`
3. **Write tests first** in `tests/unit/test_{component}.py`
4. **Implement component** following existing patterns
5. **Verify**:
   ```bash
   pytest tests/unit/test_{component}.py -v
   mypy src/specagent/retrieval/{component}.py
   ```

## Component Specifications

### Chunker (`chunker.py`)
- Use `langchain.text_splitter.RecursiveCharacterTextSplitter`
- Preserve markdown section headers
- Extract spec_id from filename (e.g., "TS38.321.md" → "TS38.321")
- Default chunk_size=512, overlap=64

### Embeddings (`embeddings.py`)
- Use `httpx` for HuggingFace Inference API calls
- Implement retry with exponential backoff (use `tenacity`)
- Batch requests (default batch_size=32)
- Normalize vectors for cosine similarity

### Indexer (`indexer.py`)
- Use `faiss.IndexFlatIP` for cosine similarity
- Store chunk metadata in parallel list
- Implement save/load for persistence
- Memory-map large indexes if needed

## Dependencies
Only use packages already in pyproject.toml:
- httpx, tenacity, numpy, faiss-cpu, langchain
