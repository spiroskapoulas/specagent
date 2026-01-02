"""
Evaluation metrics using RAGAS and custom implementations.

Metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved chunks relevant?
    - Context Recall: Did we retrieve all needed information?
    - Retrieval metrics: Recall@k, MRR
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float | None = None  # Requires ground truth
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.25,
            "context_precision": 0.2,
            "context_recall": 0.25,
        }
        return (
            self.faithfulness * weights["faithfulness"]
            + self.answer_relevancy * weights["answer_relevancy"]
            + self.context_precision * weights["context_precision"]
            + self.context_recall * weights["context_recall"]
        )


@dataclass
class RetrievalMetrics:
    """Results from retrieval evaluation."""

    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    
    @property
    def summary(self) -> dict[str, float]:
        """Return metrics as dictionary."""
        return {
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
        }


def evaluate_e2e(
    test_dataset: list[dict[str, Any]],
    include_correctness: bool = False,
) -> EvaluationResult:
    """
    Run end-to-end RAGAS evaluation.

    Args:
        test_dataset: List of dicts with keys:
            - question: str
            - answer: str (generated)
            - contexts: list[str] (retrieved chunks)
            - ground_truth: str (optional, for correctness)
        include_correctness: Whether to compute answer correctness

    Returns:
        EvaluationResult with all metric scores
    """
    # TODO: Implement RAGAS evaluation
    # 1. Format dataset for RAGAS
    # 2. Run evaluation with selected metrics
    # 3. Handle API rate limits
    # 4. Return aggregated results
    raise NotImplementedError("E2E evaluation not yet implemented")


def evaluate_retrieval(
    queries: list[str],
    retrieved_docs: list[list[str]],
    ground_truth_docs: list[list[str]],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """
    Evaluate retrieval quality.

    Args:
        queries: List of query strings
        retrieved_docs: List of retrieved doc IDs per query
        ground_truth_docs: List of relevant doc IDs per query
        k_values: Values of k for Recall@k (default: [5, 10])

    Returns:
        RetrievalMetrics with Recall@k and MRR
    """
    # TODO: Implement retrieval evaluation
    # 1. Compute Recall@k for each k value
    # 2. Compute Mean Reciprocal Rank
    # 3. Return aggregated metrics
    raise NotImplementedError("Retrieval evaluation not yet implemented")


def calculate_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """
    Calculate Recall@k for a single query.

    Args:
        retrieved: Ordered list of retrieved doc IDs
        relevant: Set of relevant doc IDs
        k: Number of top results to consider

    Returns:
        Recall score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    return len(top_k & relevant_set) / len(relevant_set)


def calculate_mrr(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.

    Args:
        retrieved: Ordered list of retrieved doc IDs
        relevant: Set of relevant doc IDs

    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    relevant_set = set(relevant)
    
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    
    return 0.0
