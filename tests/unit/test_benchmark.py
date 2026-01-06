"""
Unit tests for benchmark runner.

Tests for:
    - Loading benchmark dataset from JSON
    - Running benchmark evaluation
    - Computing accuracy metrics by difficulty
    - Generating markdown reports
    - LLM-as-judge answer comparison
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from specagent.evaluation.benchmark import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkResult,
    check_answer_correctness,
    load_benchmark_questions,
    run_benchmark,
)
from specagent.graph.state import GraphState


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_tspec_dataset(tmp_path: Path) -> Path:
    """Create sample TSpec-LLM format dataset."""
    data = {
        "question_1": {
            "question": "What is the maximum number of HARQ processes for NR?",
            "option_1": "8",
            "option_2": "16",
            "option_3": "32",
            "option_4": "64",
            "answer": "option_2: 16",
            "explanation": "The maximum number of HARQ processes is 16 for both FDD and TDD.",
            "category": "3GPP TR 38.321",
            "difficulty": "Easy"
        },
        "question_2": {
            "question": "What timer is started upon detection of radio link failure?",
            "option_1": "T300",
            "option_2": "T301",
            "option_3": "T310",
            "option_4": "T311",
            "answer": "option_4: T311",
            "explanation": "Timer T311 is started upon detection of radio link failure.",
            "category": "3GPP TR 38.331",
            "difficulty": "Intermediate"
        },
        "question_3": {
            "question": "What is the frequency range for FR1 in 5G NR?",
            "option_1": "410-7125 MHz",
            "option_2": "24250-52600 MHz",
            "option_3": "1-6 GHz",
            "option_4": "Above 24 GHz",
            "answer": "option_1: 410-7125 MHz",
            "explanation": "FR1 covers frequencies from 410 MHz to 7125 MHz.",
            "category": "3GPP TR 38.101-1",
            "difficulty": "Hard"
        }
    }

    dataset_path = tmp_path / "benchmark.json"
    with open(dataset_path, "w") as f:
        json.dump(data, f, indent=2)

    return dataset_path


@pytest.fixture
def mock_graph_response():
    """Mock graph state response for benchmarking."""
    def _create_response(question: str, answer: str, latency: float = 1500.0) -> GraphState:
        return GraphState(
            question=question,
            route_decision="retrieve",
            route_reasoning="This is a 3GPP specification question",
            generation=answer,
            citations=[],
            retrieved_chunks=[],
            graded_chunks=[],
            rewrite_count=0,
            processing_time_ms=latency,
            average_confidence=0.85,
            hallucination_check="grounded",
            ungrounded_claims=[],
            error=None,
        )
    return _create_response


# =============================================================================
# Test load_benchmark_questions
# =============================================================================


def test_load_benchmark_questions_success(sample_tspec_dataset):
    """Test loading questions from TSpec-LLM format JSON."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    assert len(questions) == 3
    assert all(isinstance(q, BenchmarkQuestion) for q in questions)

    # Check first question
    q1 = questions[0]
    assert q1.id == "question_1"
    assert "HARQ processes" in q1.question
    assert q1.answer == "16"  # Parsed from "option_2: 16"
    assert q1.difficulty == "Easy"
    assert q1.category == "3GPP TR 38.321"
    assert "option_2" in q1.correct_option


def test_load_benchmark_questions_parsing_answer_format(sample_tspec_dataset):
    """Test that answer format 'option_X: text' is correctly parsed."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    # Check all answers are extracted correctly
    assert questions[0].answer == "16"
    assert questions[1].answer == "T311"
    assert questions[2].answer == "410-7125 MHz"

    # Check option numbers are preserved
    assert questions[0].correct_option == "option_2"
    assert questions[1].correct_option == "option_4"
    assert questions[2].correct_option == "option_1"


def test_load_benchmark_questions_empty_file(tmp_path):
    """Test loading from empty JSON file."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}")

    questions = load_benchmark_questions(empty_file)
    assert len(questions) == 0


def test_load_benchmark_questions_missing_file():
    """Test loading from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_questions("/nonexistent/path.json")


# =============================================================================
# Test check_answer_correctness
# =============================================================================


def test_check_answer_correctness_exact_match():
    """Test exact string match."""
    assert check_answer_correctness("16", "16", use_llm_judge=False)
    assert check_answer_correctness("T311", "T311", use_llm_judge=False)


def test_check_answer_correctness_case_insensitive():
    """Test case-insensitive matching."""
    assert check_answer_correctness("t311", "T311", use_llm_judge=False)
    assert check_answer_correctness("T311", "t311", use_llm_judge=False)


def test_check_answer_correctness_whitespace():
    """Test whitespace normalization."""
    assert check_answer_correctness("  16  ", "16", use_llm_judge=False)
    assert check_answer_correctness("16", "  16  ", use_llm_judge=False)


def test_check_answer_correctness_fuzzy_match():
    """Test fuzzy matching for similar strings."""
    # Should match "16 processes" with "16"
    assert check_answer_correctness(
        "The answer is 16 HARQ processes",
        "16",
        use_llm_judge=False
    )

    # Should match when answer is embedded in response
    assert check_answer_correctness(
        "Timer T311 is used for this purpose",
        "T311",
        use_llm_judge=False
    )


def test_check_answer_correctness_incorrect():
    """Test detection of incorrect answers."""
    assert not check_answer_correctness("8", "16", use_llm_judge=False)
    assert not check_answer_correctness("T310", "T311", use_llm_judge=False)


@pytest.mark.unit
def test_check_answer_correctness_with_llm_judge():
    """Test LLM-as-judge for semantic matching."""
    with patch("specagent.evaluation.benchmark.llm_judge_answer") as mock_judge:
        mock_judge.return_value = True

        # Use an answer that won't match fuzzy matching to force LLM judge
        result = check_answer_correctness(
            "The system uses a different approach",
            "16",
            use_llm_judge=True
        )

        assert result is True
        mock_judge.assert_called_once()


# =============================================================================
# Test run_benchmark
# =============================================================================


def test_run_benchmark_basic(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test basic benchmark execution."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    # Mock the run_query function to return correct answers
    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        mock_run_query.side_effect = [
            mock_graph_response("Q1", "16", 1200.0),
            mock_graph_response("Q2", "T311", 1500.0),
            mock_graph_response("Q3", "410-7125 MHz", 2000.0),
        ]

        report = run_benchmark(
            questions=questions,
            limit=None,
            output_dir=tmp_path / "results"
        )

        assert report.total_questions == 3
        assert report.correct_answers == 3
        assert report.accuracy == 1.0
        assert len(report.results) == 3


def test_run_benchmark_with_limit(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test benchmark with question limit."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        mock_run_query.return_value = mock_graph_response("Q", "16", 1000.0)

        report = run_benchmark(
            questions=questions,
            limit=2,
            output_dir=tmp_path / "results"
        )

        assert report.total_questions == 2
        assert len(report.results) == 2


def test_run_benchmark_accuracy_by_difficulty(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test accuracy computation by difficulty level."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        # Return correct for Easy, incorrect for Intermediate and Hard
        mock_run_query.side_effect = [
            mock_graph_response("Q1", "16", 1000.0),      # Easy - correct
            mock_graph_response("Q2", "T310", 1000.0),    # Intermediate - wrong
            mock_graph_response("Q3", "Wrong", 1000.0),   # Hard - wrong
        ]

        report = run_benchmark(
            questions=questions,
            limit=None,
            output_dir=tmp_path / "results"
        )

        assert report.accuracy_by_difficulty["Easy"] == 1.0
        assert report.accuracy_by_difficulty["Intermediate"] == 0.0
        assert report.accuracy_by_difficulty["Hard"] == 0.0


def test_run_benchmark_saves_results(sample_tspec_dataset, mock_graph_response, tmp_path):
    """Test that benchmark saves results to JSON and markdown."""
    questions = load_benchmark_questions(sample_tspec_dataset)
    output_dir = tmp_path / "results"

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        mock_run_query.return_value = mock_graph_response("Q", "16", 1000.0)

        report = run_benchmark(
            questions=questions,
            limit=1,
            output_dir=output_dir
        )

        # Check that output directory was created
        assert output_dir.exists()

        # Check that JSON file was created
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1

        # Check that markdown file was created
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1


def test_run_benchmark_handles_errors(sample_tspec_dataset, tmp_path):
    """Test that benchmark handles errors gracefully."""
    questions = load_benchmark_questions(sample_tspec_dataset)

    with patch("specagent.graph.workflow.run_query") as mock_run_query:
        # Simulate an error in the pipeline
        error_state = GraphState(
            question="Q1",
            error="Pipeline failed",
            route_decision="reject",
            processing_time_ms=100.0,
            generation=None,
        )
        mock_run_query.return_value = error_state

        report = run_benchmark(
            questions=questions,
            limit=1,
            output_dir=tmp_path / "results"
        )

        # Should still generate report with error recorded
        assert report.total_questions == 1
        assert report.results[0].error == "Pipeline failed"
        assert not report.results[0].is_correct


# =============================================================================
# Test BenchmarkReport
# =============================================================================


def test_benchmark_report_to_dict():
    """Test conversion of report to dictionary."""
    result = BenchmarkResult(
        question_id="q1",
        question="Test?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
        rewrites=0,
        error=None,
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=1,
        correct_answers=1,
        accuracy=1.0,
        accuracy_by_difficulty={"Easy": 1.0},
        average_latency_ms=1500.0,
        average_confidence=0.85,
        results=[result],
    )

    report_dict = report.to_dict()

    assert report_dict["total_questions"] == 1
    assert report_dict["accuracy"] == 1.0
    assert len(report_dict["results"]) == 1
    assert report_dict["results"][0]["is_correct"] is True


def test_benchmark_report_to_markdown():
    """Test markdown report generation."""
    result1 = BenchmarkResult(
        question_id="q1",
        question="What is X?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
    )

    result2 = BenchmarkResult(
        question_id="q2",
        question="What is Y?",
        expected_answer="T311",
        generated_answer="T310",
        is_correct=False,
        confidence=0.70,
        latency_ms=2000.0,
        difficulty="Hard",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=2,
        correct_answers=1,
        accuracy=0.5,
        accuracy_by_difficulty={"Easy": 1.0, "Hard": 0.0},
        average_latency_ms=1750.0,
        average_confidence=0.775,
        results=[result1, result2],
    )

    markdown = report.to_markdown()

    # Check header
    assert "# SpecAgent Benchmark Report" in markdown

    # Check summary table
    assert "Total Questions | 2" in markdown
    assert "Correct Answers | 1" in markdown
    assert "50.0%" in markdown

    # Check difficulty breakdown
    assert "Easy" in markdown
    assert "Hard" in markdown

    # Check failed questions section
    assert "Failed Questions" in markdown
    assert "q2" in markdown
    assert "What is Y?" in markdown


def test_benchmark_report_markdown_no_failures():
    """Test markdown report when all questions pass."""
    result = BenchmarkResult(
        question_id="q1",
        question="Test?",
        expected_answer="16",
        generated_answer="16",
        is_correct=True,
        confidence=0.85,
        latency_ms=1500.0,
        difficulty="Easy",
    )

    report = BenchmarkReport(
        timestamp="2024-01-01T12:00:00",
        total_questions=1,
        correct_answers=1,
        accuracy=1.0,
        accuracy_by_difficulty={"Easy": 1.0},
        average_latency_ms=1500.0,
        average_confidence=0.85,
        results=[result],
    )

    markdown = report.to_markdown()

    assert "No failed questions!" in markdown


# =============================================================================
# Test BenchmarkQuestion
# =============================================================================


def test_benchmark_question_creation():
    """Test BenchmarkQuestion dataclass."""
    question = BenchmarkQuestion(
        id="q1",
        question="What is the answer?",
        answer="42",
        difficulty="medium",
        category="Test Category",
        correct_option="option_1",
        spec_references=["TS38.321"],
    )

    assert question.id == "q1"
    assert question.answer == "42"
    assert question.difficulty == "medium"
