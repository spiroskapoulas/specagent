# Benchmark Runner

This directory contains the benchmark evaluation script for SpecAgent.

## Usage

### Basic Usage

Run the benchmark on the full TSpec-LLM dataset:

```bash
python scripts/run_benchmark.py --dataset data/qna/Sampled_3GPP_TR_Questions.json
```

### Test with Limited Questions

For quick testing, limit the number of questions:

```bash
# Test with first 10 questions
python scripts/run_benchmark.py --limit 10

# Test with first 5 questions
python scripts/run_benchmark.py --limit 5
```

### Custom Output Directory

Save results to a custom directory:

```bash
python scripts/run_benchmark.py --output-dir evaluation/results/2024-01-01
```

## Output

The benchmark runner generates two files:

1. **JSON Report** (`benchmark_<timestamp>.json`): Machine-readable results with detailed metrics
2. **Markdown Report** (`benchmark_<timestamp>.md`): Human-readable report with:
   - Summary table (total questions, accuracy, latency, confidence)
   - Accuracy breakdown by difficulty (Easy, Intermediate, Hard)
   - Failed questions with expected vs. generated answers

## Example Output

```
SpecAgent Benchmark Runner

Dataset:     data/qna/Sampled_3GPP_TR_Questions.json
Output Dir:  evaluation/results
Limit:       10 questions

Loading benchmark questions...
Loaded 100 questions
Limited to 10 questions

Running benchmark evaluation...

┌─────────────────────────────────────────────┐
│               Benchmark Results             │
└─────────────────────────────────────────────┘

Summary
┌──────────────────────┬─────────┐
│ Metric               │   Value │
├──────────────────────┼─────────┤
│ Total Questions      │      10 │
│ Correct Answers      │       8 │
│ Overall Accuracy     │  80.0%  │
│ Average Latency      │ 1500ms  │
│ Average Confidence   │    0.82 │
└──────────────────────┴─────────┘

Accuracy by Difficulty
┌─────────────────┬──────────┬───────────┐
│ Difficulty      │ Accuracy │ Questions │
├─────────────────┼──────────┼───────────┤
│ Easy            │  100.0%  │         4 │
│ Intermediate    │   75.0%  │         4 │
│ Hard            │   50.0%  │         2 │
└─────────────────┴──────────┴───────────┘

Output Files
JSON Report:  evaluation/results/benchmark_2024-01-01T12-00-00.json
MD Report:    evaluation/results/benchmark_2024-01-01T12-00-00.md
```

## Dataset Format

The benchmark script expects TSpec-LLM format JSON:

```json
{
  "question_1": {
    "question": "What is the maximum number of HARQ processes for NR?",
    "option_1": "8",
    "option_2": "16",
    "option_3": "32",
    "option_4": "64",
    "answer": "option_2: 16",
    "explanation": "...",
    "category": "3GPP TR 38.321",
    "difficulty": "Easy"
  },
  ...
}
```

## Accuracy Checking

The benchmark uses a multi-stage answer checking approach:

1. **Exact Match**: Case-insensitive exact string match
2. **Fuzzy Match**: Check if expected answer is contained in generated answer
3. **Word-Based Match**: Check if all expected words are present
4. **LLM Judge**: Use LLM for semantic comparison (optional, enabled by default)

## Performance Targets

- **Target Accuracy**: 85%+ (baseline naive RAG: 71-75%)
- **Latency Target**: <3 seconds P95
- **Breakdown**: Track accuracy by difficulty level (Easy/Intermediate/Hard)
