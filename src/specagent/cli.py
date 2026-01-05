"""
Command-line interface for SpecAgent.

Commands:
    serve     - Start the FastAPI server
    query     - Run a single query
    index     - Build or update the FAISS index
    benchmark - Run evaluation benchmark
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="specagent",
    help="Agentic RAG for 3GPP specifications",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[green]Starting SpecAgent server on {host}:{port}[/green]")
    
    uvicorn.run(
        "specagent.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,  # Memory constraint
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run a single query through the pipeline."""
    from specagent.graph.workflow import run_query

    console.print(f"[blue]Question:[/blue] {question}\n")

    with console.status("[bold green]Processing..."):
        result = run_query(question)

    # Check for rejection
    if result.get("route_decision") == "reject":
        console.print("[yellow]This question is outside 3GPP specifications.[/yellow]")
        if verbose:
            console.print(f"[dim]Reasoning: {result.get('route_reasoning', 'N/A')}[/dim]")
        return

    # Display answer
    console.print("[green]Answer:[/green]")
    console.print(result.get("generation", "No answer generated."))
    console.print()

    # Display citations
    citations = result.get("citations", [])
    if citations:
        console.print("[blue]Citations:[/blue]")
        for c in citations:
            console.print(f"  • {c.raw_citation}")
        console.print()

    # Display metadata
    if verbose:
        table = Table(title="Metadata")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Latency", f"{result.get('processing_time_ms', 0):.0f}ms")
        table.add_row("Chunks Retrieved", str(len(result.get("retrieved_chunks", []))))
        table.add_row("Rewrites", str(result.get("rewrite_count", 0)))
        table.add_row("Hallucination Check", result.get("hallucination_check", "N/A"))
        table.add_row("Confidence", f"{result.get('average_confidence', 0):.2f}")

        console.print(table)


@app.command()
def index(
    data_dir: str = typer.Option("data/raw", help="Directory with markdown files"),
    output_dir: str = typer.Option("data/index", help="Directory for index files"),
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if exists"),
    download: bool = typer.Option(False, "--download", "-d", help="Download specs from HuggingFace (requires license acceptance)"),
    use_git: bool = typer.Option(False, "--use-git", help="Use git clone instead of HTTP download"),
) -> None:
    """Build or update the FAISS index."""
    from pathlib import Path

    import numpy as np

    from specagent.config import settings
    from specagent.retrieval.chunker import chunk_markdown
    from specagent.retrieval.data_ingestion import (
        clone_dataset_with_git_lfs,
        discover_markdown_files,
        download_38_series_specs,
    )
    from specagent.retrieval.embeddings import LocalEmbedder
    from specagent.retrieval.indexer import FAISSIndex

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Download files if requested
    if download:
        console.print("[cyan]Downloading specifications from HuggingFace...[/cyan]")
        console.print("[yellow]⚠ This dataset is GATED and requires license acceptance![/yellow]")
        console.print("[dim]1. Visit: https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM[/dim]")
        console.print("[dim]2. Log in and accept the CC-BY-NC-4.0 license[/dim]")
        console.print("[dim]3. Make sure your HF_API_KEY has access to this dataset[/dim]\n")

        try:
            if use_git:
                console.print("[cyan]Using git clone method...[/cyan]")
                downloaded = clone_dataset_with_git_lfs(data_path)
            else:
                downloaded = download_38_series_specs(data_path)

            if not downloaded:
                console.print("[red]No files downloaded. Exiting.[/red]")
                console.print("[yellow]If you're getting 401 errors, you may need to:[/yellow]")
                console.print("  1. Accept the dataset license on HuggingFace")
                console.print("  2. Use --use-git flag to clone with git credentials")
                console.print("  3. Or manually download files to data/raw/ and run without --download")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            console.print("\n[yellow]Alternative: Manually download the dataset:[/yellow]")
            console.print("  git clone https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM")
            console.print("  cp TSpec-LLM/3GPP-clean/Rel-18/38_series/*.md data/raw/")
            console.print("  specagent index --force")
            raise typer.Exit(1)

    # Validate data directory exists
    if not data_path.exists():
        console.print(f"[red]Data directory not found: {data_path}[/red]")
        console.print("Use --download to fetch specifications from HuggingFace.")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "faiss.index"
    if index_file.exists() and not force:
        console.print(f"[yellow]Index already exists: {index_file}[/yellow]")
        console.print("Use --force to rebuild.")
        return

    console.print(f"[blue]Building index from {data_path}...[/blue]\n")

    # Step 1: Load markdown files
    console.print("[cyan]Step 1: Discovering markdown files...[/cyan]")
    md_files = discover_markdown_files(data_path)
    if not md_files:
        console.print(f"[red]No markdown files found in {data_path}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Found {len(md_files)} markdown files[/green]\n")

    # Step 2: Chunk documents
    console.print("[cyan]Step 2: Chunking documents...[/cyan]")
    file_chunks_map = []  # List of (md_file, chunks) tuples
    total_chars = 0

    with console.status("[bold green]Chunking...") as status:
        for md_file in md_files:
            status.update(f"[bold green]Chunking {md_file.name}...")

            text = md_file.read_text(encoding="utf-8")
            total_chars += len(text)

            chunks = chunk_markdown(
                text=text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            # Update source_file metadata
            for chunk in chunks:
                chunk.metadata["source_file"] = md_file.name

            file_chunks_map.append((md_file, chunks))
            console.print(f"[green]  ✓ {md_file.name}: {len(chunks)} chunks ({len(text):,} chars)[/green]")

    total_chunks = sum(len(chunks) for _, chunks in file_chunks_map)
    console.print(f"[green]Created {total_chunks} chunks from {total_chars:,} characters[/green]\n")

    # Step 3: Generate embeddings
    console.print("[cyan]Step 3: Generating embeddings...[/cyan]")
    console.print(f"[dim]Using model: {settings.embedding_model} (local)[/dim]\n")

    embedder = LocalEmbedder()
    all_chunks = []
    all_embeddings = []

    for idx, (md_file, chunks) in enumerate(file_chunks_map, 1):
        console.print(f"[blue]({idx}/{len(file_chunks_map)}) Embedding {md_file.name}[/blue] - {len(chunks)} chunks")

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedder.embed_texts(chunk_texts)

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

        console.print(f"[green]  ✓ Generated {len(embeddings)} embeddings[/green]\n")

    embeddings = np.vstack(all_embeddings)
    console.print(f"[green]Total: {len(embeddings)} embeddings with dimension {embeddings.shape[1]}[/green]\n")

    # Step 4: Build FAISS index
    console.print("[cyan]Step 4: Building FAISS index...[/cyan]")
    faiss_index = FAISSIndex(dimension=settings.embedding_dimension)

    with console.status("[bold green]Building index..."):
        faiss_index.build(all_chunks, embeddings)

    console.print(f"[green]Built index with {faiss_index.size} vectors[/green]\n")

    # Step 5: Save to output directory
    console.print("[cyan]Step 5: Saving index...[/cyan]")
    index_path = output_path / "faiss"

    with console.status(f"[bold green]Saving to {index_path}..."):
        faiss_index.save(index_path)

    console.print(f"[green]Saved index to {index_path}.index and {index_path}.json[/green]\n")

    # Summary
    console.print("[bold green]✓ Index building complete![/bold green]")
    console.print(f"  Files processed: {len(md_files)}")
    console.print(f"  Total chunks: {len(all_chunks)}")
    console.print(f"  Index size: {faiss_index.size} vectors")
    console.print(f"  Output: {index_path}.{{index,json}}")


@app.command()
def benchmark(
    dataset: str = typer.Option(
        "data/evaluation/tspec_benchmark.json",
        help="Path to benchmark dataset",
    ),
    output_dir: str = typer.Option(
        "evaluation/results",
        help="Directory for results",
    ),
    limit: int = typer.Option(None, help="Limit number of questions"),
) -> None:
    """Run evaluation benchmark."""
    from pathlib import Path

    from specagent.evaluation.benchmark import (
        load_benchmark_questions,
        run_benchmark,
    )

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading benchmark from {dataset_path}...[/blue]")
    questions = load_benchmark_questions(dataset_path)

    if limit:
        questions = questions[:limit]
        console.print(f"[yellow]Limited to {limit} questions[/yellow]")

    console.print(f"[blue]Running {len(questions)} questions...[/blue]")

    # TODO: Implement benchmark execution
    console.print("[red]Benchmark runner not yet implemented.[/red]")


@app.command()
def version() -> None:
    """Show version information."""
    from specagent import __version__

    console.print(f"SpecAgent v{__version__}")


if __name__ == "__main__":
    app()
