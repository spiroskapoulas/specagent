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
    answer: str  # Ground truth (parsed from "option_X: text")
    difficulty: str  # Easy, Intermediate, Hard
    correct_option: str = ""  # e.g., "option_2"
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
    Load benchmark questions from TSpec-LLM format JSON file.

    Expected format:
    {
        "question_1": {
            "question": "...",
            "option_1": "...",
            "option_2": "...",
            "option_3": "...",
            "option_4": "...",
            "answer": "option_2: text",
            "explanation": "...",
            "category": "3GPP TR ...",
            "difficulty": "Easy|Intermediate|Hard"
        },
        ...
    }

    Args:
        path: Path to benchmark JSON file

    Returns:
        List of BenchmarkQuestion objects
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    questions = []
    for question_id, q_data in data.items():
        # Parse answer format: "option_2: 16" -> answer="16", correct_option="option_2"
        answer_text = q_data["answer"]
        if ":" in answer_text:
            correct_option, answer = answer_text.split(":", 1)
            correct_option = correct_option.strip()
            answer = answer.strip()
        else:
            correct_option = ""
            answer = answer_text

        questions.append(
            BenchmarkQuestion(
                id=question_id,
                question=q_data["question"],
                answer=answer,
                difficulty=q_data.get("difficulty", "Intermediate"),
                correct_option=correct_option,
                spec_references=[],
                category=q_data.get("category", ""),
            )
        )

    return questions


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
    from specagent.graph.workflow import run_query

    # Apply limit if specified
    if limit is not None:
        questions = questions[:limit]

    # Initialize results
    results: list[BenchmarkResult] = []

    # Run each question through the pipeline
    for question in questions:
        try:
            # Execute pipeline
            state = run_query(question.question)

            # Extract generated answer
            generated_answer = state.get("generation", "")

            # Handle errors or rejections
            if state.get("error"):
                results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        expected_answer=question.answer,
                        generated_answer=generated_answer or "",
                        is_correct=False,
                        confidence=0.0,
                        latency_ms=state.get("processing_time_ms", 0.0),
                        difficulty=question.difficulty,
                        rewrites=state.get("rewrite_count", 0),
                        error=state.get("error"),
                    )
                )
                continue

            if state.get("route_decision") == "reject":
                results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        expected_answer=question.answer,
                        generated_answer="",
                        is_correct=False,
                        confidence=0.0,
                        latency_ms=state.get("processing_time_ms", 0.0),
                        difficulty=question.difficulty,
                        rewrites=state.get("rewrite_count", 0),
                        error="Question was rejected by router",
                    )
                )
                continue

            # Check correctness
            is_correct = check_answer_correctness(
                generated_answer,
                question.answer,
                use_llm_judge=True,
            )

            results.append(
                BenchmarkResult(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.answer,
                    generated_answer=generated_answer,
                    is_correct=is_correct,
                    confidence=state.get("average_confidence", 0.0),
                    latency_ms=state.get("processing_time_ms", 0.0),
                    difficulty=question.difficulty,
                    rewrites=state.get("rewrite_count", 0),
                    error=None,
                )
            )

        except Exception as e:
            # Handle unexpected errors
            results.append(
                BenchmarkResult(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.answer,
                    generated_answer="",
                    is_correct=False,
                    confidence=0.0,
                    latency_ms=0.0,
                    difficulty=question.difficulty,
                    rewrites=0,
                    error=str(e),
                )
            )

    # Compute metrics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.is_correct)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0

    # Accuracy by difficulty
    accuracy_by_difficulty: dict[str, float] = {}
    for difficulty in ["Easy", "Intermediate", "Hard"]:
        difficulty_results = [r for r in results if r.difficulty == difficulty]
        if difficulty_results:
            correct = sum(1 for r in difficulty_results if r.is_correct)
            accuracy_by_difficulty[difficulty] = correct / len(difficulty_results)

    # Average metrics
    average_latency_ms = (
        sum(r.latency_ms for r in results) / total_questions
        if total_questions > 0
        else 0.0
    )
    average_confidence = (
        sum(r.confidence for r in results) / total_questions
        if total_questions > 0
        else 0.0
    )

    # Create report
    timestamp = datetime.now().isoformat()
    report = BenchmarkReport(
        timestamp=timestamp,
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy=accuracy,
        accuracy_by_difficulty=accuracy_by_difficulty,
        average_latency_ms=average_latency_ms,
        average_confidence=average_confidence,
        results=results,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_filename = f"benchmark_{timestamp.replace(':', '-').split('.')[0]}.json"
    json_path = output_path / json_filename
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Save Markdown
    md_filename = f"benchmark_{timestamp.replace(':', '-').split('.')[0]}.md"
    md_path = output_path / md_filename
    with open(md_path, "w") as f:
        f.write(report.to_markdown())

    return report


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
    # Normalize both strings
    generated_norm = generated.strip().lower()
    expected_norm = expected.strip().lower()

    # 1. Try exact match
    if generated_norm == expected_norm:
        return True

    # 2. Try fuzzy string matching - check if expected is contained in generated
    # This handles cases like "The answer is 16" containing "16"
    if expected_norm in generated_norm:
        return True

    # 3. Try word-based matching for multi-word answers
    expected_words = set(expected_norm.split())
    generated_words = set(generated_norm.split())

    # If all expected words are in generated, consider it a match
    if expected_words and expected_words.issubset(generated_words):
        return True

    # 4. Use LLM judge for semantic comparison if requested
    if use_llm_judge:
        return llm_judge_answer(generated, expected)

    return False


def llm_judge_answer(generated: str, expected: str) -> bool:
    """
    Use LLM to judge if generated answer is semantically equivalent to expected.

    Args:
        generated: Generated answer
        expected: Expected answer

    Returns:
        True if answers are semantically equivalent
    """
    from specagent.llm.factory import get_llm

    prompt = f"""You are evaluating answers to technical questions about 3GPP specifications.

Question: Does the generated answer convey the same information as the expected answer?

Expected Answer: {expected}
Generated Answer: {generated}

Respond with ONLY "yes" or "no".

If the generated answer contains the expected answer or conveys the same key information (even with additional context), respond "yes".
If the generated answer is incorrect, contradicts the expected answer, or is missing the key information, respond "no".

Response:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        # Extract text from response
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        # Check if response contains "yes"
        return "yes" in response_text.lower().strip()

    except Exception:
        # If LLM judge fails, fall back to fuzzy matching
        return expected.lower() in generated.lower()
