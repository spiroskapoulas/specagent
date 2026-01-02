#!/usr/bin/env python3
"""
Download TSpec-LLM dataset from HuggingFace.

Requires HuggingFace authentication:
    huggingface-cli login

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --release 18 --output-dir data/raw
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from rich.console import Console
from rich.progress import Progress

console = Console()

DATASET_REPO = "rasoul-nikbakht/TSpec-LLM"


def download_tspec_llm(
    release: int = 18,
    output_dir: str = "data/raw",
    subset: str = "3GPP-clean",
) -> None:
    """
    Download TSpec-LLM markdown files.

    Args:
        release: 3GPP release number (e.g., 18)
        output_dir: Output directory for downloaded files
        subset: Dataset subset ("3GPP-clean" or "3GPP-raw")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Downloading TSpec-LLM Release {release}...[/blue]")
    console.print(f"Repository: {DATASET_REPO}")
    console.print(f"Output: {output_path.absolute()}")

    try:
        # List files in the repository
        files = list_repo_files(DATASET_REPO, repo_type="dataset")

        # Filter for the requested release
        pattern = f"{subset}/Rel-{release}/"
        matching_files = [f for f in files if pattern in f and f.endswith(".md")]

        if not matching_files:
            console.print(f"[red]No files found for pattern: {pattern}[/red]")
            console.print("Available files sample:", files[:10])
            return

        console.print(f"[green]Found {len(matching_files)} markdown files[/green]")

        # Download files with progress bar
        with Progress() as progress:
            task = progress.add_task("Downloading...", total=len(matching_files))

            for file_path in matching_files:
                # Download file
                local_path = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=output_path,
                )
                progress.advance(task)

        console.print(f"[green]✓ Downloaded {len(matching_files)} files[/green]")
        console.print(f"[blue]Files saved to: {output_path.absolute()}[/blue]")

    except Exception as e:
        console.print(f"[red]Error downloading dataset: {e}[/red]")
        console.print("[yellow]Make sure you're logged in: huggingface-cli login[/yellow]")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download TSpec-LLM dataset from HuggingFace"
    )
    parser.add_argument(
        "--release",
        type=int,
        default=18,
        help="3GPP release number (default: 18)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="3GPP-clean",
        choices=["3GPP-clean", "3GPP-raw"],
        help="Dataset subset (default: 3GPP-clean)",
    )

    args = parser.parse_args()
    download_tspec_llm(
        release=args.release,
        output_dir=args.output_dir,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
