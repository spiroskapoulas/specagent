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
) -> None:
    """Build or update the FAISS index."""
    from pathlib import Path

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if not data_path.exists():
        console.print(f"[red]Data directory not found: {data_path}[/red]")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "faiss.index"
    if index_file.exists() and not force:
        console.print(f"[yellow]Index already exists: {index_file}[/yellow]")
        console.print("Use --force to rebuild.")
        return

    console.print(f"[blue]Building index from {data_path}...[/blue]")

    # TODO: Implement index building
    # 1. Load markdown files
    # 2. Chunk documents
    # 3. Generate embeddings
    # 4. Build FAISS index
    # 5. Save to output directory

    console.print("[red]Index building not yet implemented.[/red]")


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
