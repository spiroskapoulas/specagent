"""
TSpec-LLM benchmark runner.

Runs the SpecAgent pipeline against the TSpec-LLM benchmark
questions and computes accuracy by difficulty level.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkQuestion:
    """A question from the TSpec-LLM benchmark."""

    id: str
    question: str
    answer: str  # Ground truth
    difficulty: str  # easy, medium, hard
    spec_references: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class BenchmarkResult:
    """Result for a single benchmark question."""

    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    is_correct: bool
    confidence: float
    latency_ms: float
    difficulty: str
    rewrites: int = 0
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    timestamp: str
    total_questions: int
    correct_answers: int
    accuracy: float
    accuracy_by_difficulty: dict[str, float]
    average_latency_ms: float
    average_confidence: float
    results: list[BenchmarkResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "average_latency_ms": self.average_latency_ms,
            "average_confidence": self.average_confidence,
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "generated_answer": r.generated_answer,
                    "is_correct": r.is_correct,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "difficulty": r.difficulty,
                    "rewrites": r.rewrites,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# SpecAgent Benchmark Report",
            f"",
            f"**Date:** {self.timestamp}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Questions | {self.total_questions} |",
            f"| Correct Answers | {self.correct_answers} |",
            f"| **Accuracy** | **{self.accuracy:.1%}** |",
            f"| Average Latency | {self.average_latency_ms:.0f}ms |",
            f"| Average Confidence | {self.average_confidence:.2f} |",
            f"",
            f"## Accuracy by Difficulty",
            f"",
            f"| Difficulty | Accuracy |",
            f"|------------|----------|",
        ]
        
        for difficulty, acc in sorted(self.accuracy_by_difficulty.items()):
            lines.append(f"| {difficulty.capitalize()} | {acc:.1%} |")
        
        lines.extend([
            f"",
            f"## Failed Questions",
            f"",
        ])
        
        failed = [r for r in self.results if not r.is_correct]
        if failed:
            for r in failed[:10]:  # Limit to first 10
                lines.extend([
                    f"### {r.question_id}",
                    f"",
                    f"**Question:** {r.question}",
                    f"",
                    f"**Expected:** {r.expected_answer}",
                    f"",
                    f"**Generated:** {r.generated_answer}",
                    f"",
                ])
        else:
            lines.append("No failed questions! 🎉")
        
        return "\n".join(lines)


def load_benchmark_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """
    Load benchmark questions from JSON file.

    Args:
        path: Path to benchmark JSON file

    Returns:
        List of BenchmarkQuestion objects
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    
    return [
        BenchmarkQuestion(
            id=q.get("id", f"q_{i}"),
            question=q["question"],
            answer=q["answer"],
            difficulty=q.get("difficulty", "medium"),
            spec_references=q.get("spec_references", []),
            category=q.get("category", ""),
        )
        for i, q in enumerate(data)
    ]


def run_benchmark(
    questions: list[BenchmarkQuestion],
    limit: int | None = None,
    output_dir: str | Path = "evaluation/results",
) -> BenchmarkReport:
    """
    Run benchmark evaluation.

    Args:
        questions: List of benchmark questions
        limit: Maximum number of questions to run (for testing)
        output_dir: Directory to save results

    Returns:
        BenchmarkReport with all results
    """
    # TODO: Implement benchmark runner
    # 1. For each question, run through pipeline
    # 2. Compare generated answer to ground truth
    # 3. Use LLM-as-judge for fuzzy matching
    # 4. Compute accuracy metrics
    # 5. Save results to JSON and markdown
    raise NotImplementedError("Benchmark runner not yet implemented")


def check_answer_correctness(
    generated: str,
    expected: str,
    use_llm_judge: bool = True,
) -> bool:
    """
    Check if generated answer matches expected answer.

    Uses fuzzy matching since exact string match is too strict.

    Args:
        generated: Generated answer text
        expected: Expected/ground truth answer
        use_llm_judge: Whether to use LLM for semantic comparison

    Returns:
        True if answers match, False otherwise
    """
    # TODO: Implement answer checking
    # 1. Try exact match first
    # 2. Try fuzzy string matching
    # 3. If still not matched and use_llm_judge, use LLM
    raise NotImplementedError("Answer checking not yet implemented")
