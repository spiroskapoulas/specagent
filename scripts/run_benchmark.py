#!/usr/bin/env python3
"""
Benchmark runner for SpecAgent RAG evaluation.

Loads TSpec-LLM benchmark questions, runs them through the pipeline,
computes accuracy by difficulty, and generates reports.

Usage:
    python scripts/run_benchmark.py --dataset data/qna/Sampled_3GPP_TR_Questions.json
    python scripts/run_benchmark.py --limit 10  # Test with first 10 questions
    python scripts/run_benchmark.py --output-dir results/2024-01-01
"""

import argparse
import sys
from pathlib import Path

# Add src to path to allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from specagent.evaluation.benchmark import (
    load_benchmark_questions,
    run_benchmark,
)


console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SpecAgent benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python scripts/run_benchmark.py --dataset data/qna/Sampled_3GPP_TR_Questions.json

  # Test with first 10 questions
  python scripts/run_benchmark.py --limit 10

  # Custom output directory
  python scripts/run_benchmark.py --output-dir evaluation/results/2024-01-01
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qna/Sampled_3GPP_TR_Questions.json",
        help="Path to TSpec-LLM benchmark JSON file (default: data/qna/Sampled_3GPP_TR_Questions.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory to save results (default: evaluation/results)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to run (useful for testing)",
    )

    return parser.parse_args()


def display_results(report):
    """Display benchmark results in a formatted table."""
    console.print("\n")
    console.rule("[bold blue]Benchmark Results[/bold blue]")
    console.print()

    # Summary table
    summary_table = Table(title="Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Questions", str(report.total_questions))
    summary_table.add_row("Correct Answers", str(report.correct_answers))
    summary_table.add_row("Overall Accuracy", f"{report.accuracy:.1%}")
    summary_table.add_row("Average Latency", f"{report.average_latency_ms:.0f}ms")
    summary_table.add_row("Average Confidence", f"{report.average_confidence:.2f}")

    console.print(summary_table)
    console.print()

    # Accuracy by difficulty
    if report.accuracy_by_difficulty:
        diff_table = Table(
            title="Accuracy by Difficulty",
            show_header=True,
            header_style="bold cyan",
        )
        diff_table.add_column("Difficulty", style="cyan", width=20)
        diff_table.add_column("Accuracy", style="green", justify="right")
        diff_table.add_column("Questions", style="dim", justify="right")

        for difficulty in ["Easy", "Intermediate", "Hard"]:
            if difficulty in report.accuracy_by_difficulty:
                accuracy = report.accuracy_by_difficulty[difficulty]
                count = sum(1 for r in report.results if r.difficulty == difficulty)

                # Color code based on accuracy
                if accuracy >= 0.85:
                    style = "green"
                elif accuracy >= 0.70:
                    style = "yellow"
                else:
                    style = "red"

                diff_table.add_row(
                    difficulty,
                    f"[{style}]{accuracy:.1%}[/{style}]",
                    str(count),
                )

        console.print(diff_table)
        console.print()

    # Failed questions summary
    failed = [r for r in report.results if not r.is_correct]
    if failed:
        console.print(f"[yellow]Failed Questions: {len(failed)}[/yellow]")
        console.print("[dim]See markdown report for details[/dim]")
    else:
        console.print("[bold green]✓ All questions answered correctly![/bold green]")

    console.print()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
        console.print("\nPlease provide a valid path to the benchmark dataset.")
        console.print("Example: --dataset data/qna/Sampled_3GPP_TR_Questions.json")
        sys.exit(1)

    # Display configuration
    console.print("[bold blue]SpecAgent Benchmark Runner[/bold blue]")
    console.print()
    console.print(f"Dataset:     {dataset_path}")
    console.print(f"Output Dir:  {args.output_dir}")
    if args.limit:
        console.print(f"Limit:       {args.limit} questions")
    console.print()

    # Load questions
    console.print("[cyan]Loading benchmark questions...[/cyan]")
    try:
        questions = load_benchmark_questions(dataset_path)
        total_questions = len(questions)
        console.print(f"[green]Loaded {total_questions} questions[/green]")

        if args.limit:
            console.print(f"[yellow]Will process first {args.limit} questions[/yellow]")

    except Exception as e:
        console.print(f"[red]Error loading questions: {e}[/red]")
        sys.exit(1)

    console.print()

    # Run benchmark
    console.print("[cyan]Running benchmark evaluation...[/cyan]")
    console.print("[dim]This may take several minutes depending on the number of questions[/dim]")
    console.print()

    try:
        with console.status("[bold green]Processing questions..."):
            report = run_benchmark(
                questions=questions,
                limit=args.limit,
                output_dir=args.output_dir,
            )

        # Display results
        display_results(report)

        # Show output paths
        console.rule("[bold blue]Output Files[/bold blue]")
        console.print()
        output_path = Path(args.output_dir)
        json_files = sorted(output_path.glob("benchmark_*.json"))
        md_files = sorted(output_path.glob("benchmark_*.md"))

        if json_files:
            console.print(f"[green]JSON Report:[/green]  {json_files[-1]}")
        if md_files:
            console.print(f"[green]MD Report:[/green]    {md_files[-1]}")

        console.print()

        # Summary message
        if report.accuracy >= 0.85:
            console.print("[bold green]✓ Benchmark complete! Target accuracy achieved (≥85%).[/bold green]")
        elif report.accuracy >= 0.75:
            console.print("[bold yellow]⚠ Benchmark complete. Accuracy is above baseline (71-75%) but below target (85%).[/bold yellow]")
        else:
            console.print("[bold red]✗ Benchmark complete. Accuracy is below baseline (71-75%).[/bold red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error running benchmark: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
