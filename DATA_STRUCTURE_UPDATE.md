# Data Structure Update - Multi-Release Support

## Summary

Updated the data ingestion and indexing system to support multiple 3GPP specification releases (Rel-15, Rel-16, Rel-17) organized in separate subdirectories.

## Changes Made

### 1. Updated `discover_markdown_files()` in `data_ingestion.py`

**Before:**
- Only searched for `.md` files in a single flat directory (`data/raw/*.md`)

**After:**
- Recursively searches for `.md` files in all subdirectories (`data/**/*.md`)
- Supports both legacy flat structure and new multi-release structure

### 2. Updated `specagent index` CLI command in `cli.py`

**Before:**
- Default data directory: `data/raw`
- Used `download_38_series_specs()` for downloads

**After:**
- Default data directory: `data` (searches recursively)
- Uses `download_all_required_specs()` for downloads
- Downloads all required specs across multiple releases

## Directory Structure

```
data/
├── raw_rel_15_36_series/    # Rel-15 36-series (2 files)
│   ├── 36777-f00_1.md
│   └── 36777-f00_2.md
├── raw_rel_15_38_series/    # Rel-15 38-series (1 file)
│   └── 38811-f40.md
├── raw_rel_16_38_series/    # Rel-16 38-series (2 files)
│   ├── 38821-g20.md
│   └── 38901-g10.md
├── raw_rel_17_36_series/    # Rel-17 36-series (123 files)
│   └── [123 specification files]
├── raw_rel_17_38_series/    # Rel-17 38-series (169 files)
│   └── [169 specification files]
└── index/                   # FAISS index output
    ├── faiss.index
    └── faiss.json
```

**Total: 297 specification files**

## Usage

### Manual Download (Python)

```python
from pathlib import Path
from specagent.retrieval.data_ingestion import download_all_required_specs

# Download all required specs
results = download_all_required_specs(
    base_dir=Path('data'),
    api_key='your_hf_api_key'  # Optional if HF_API_KEY env var is set
)

# Results contains:
# - rel_17_36_series: 123 files
# - rel_17_38_series: 169 files
# - rel_15_36_series: 2 files
# - rel_15_38_series: 1 file
# - rel_16_38_series: 2 files
```

### CLI Download & Index

```bash
# Set your HuggingFace API key
export HF_API_KEY="hf_..."

# Download specs and build index
specagent index --download --force

# Or just build index from existing downloaded specs
specagent index --force
```

### Discover Files

```python
from pathlib import Path
from specagent.retrieval.data_ingestion import discover_markdown_files

# Discover all markdown files recursively
md_files = discover_markdown_files(Path('data'))
# Returns: 297 files from all subdirectories
```

## Backward Compatibility

The updated code maintains backward compatibility with:
- Legacy flat directory structure (`data/raw/*.md`)
- Old single-directory downloads
- Existing indexing workflows

Files in any subdirectory under `data/` will be discovered and indexed.

## Testing

Verified:
- ✓ All 297 files are discoverable
- ✓ Files are readable from all subdirectories
- ✓ Download function creates correct directory structure
- ✓ Recursive glob pattern works correctly

## Future Usage

To manually download and index specifications in the future:

1. **Download specs:**
   ```bash
   export HF_API_KEY="your_api_key"
   python3 -c "from pathlib import Path; from specagent.retrieval.data_ingestion import download_all_required_specs; download_all_required_specs(base_dir=Path('data'))"
   ```

2. **Build index:**
   ```bash
   specagent index --force
   ```

Or use the combined command:
```bash
specagent index --download --force
```
