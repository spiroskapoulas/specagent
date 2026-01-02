# Claude Code Development Guide for 3GPP SpecAgent

## Overview

This guide outlines an efficient and safe workflow for using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to accelerate SpecAgent development. The key principles are:

1. **Decompose into isolated, testable units** — Claude Code excels at well-scoped tasks
2. **Test-first development** — Write tests before implementation to catch errors early
3. **Incremental commits** — Small, reviewable changes with clear boundaries
4. **Human review gates** — You review all generated code before merging

---

## 1. Project Initialization Strategy

### 1.1 Let Claude Code Scaffold, You Architect

**DO**: Have Claude Code generate boilerplate and structure
**DON'T**: Let Claude Code make architectural decisions without your input

```bash
# Good: Specific, bounded task
claude "Create the project structure for a Python package called specagent 
with src layout, pytest configuration, and a pyproject.toml using poetry. 
Include placeholder modules for: nodes/, retrieval/, evaluation/, api/"

# Bad: Too open-ended
claude "Build me an agentic RAG system"
```

### 1.2 Recommended Initial Scaffold

Ask Claude Code to create this structure in phases:

```
specagent/
├── pyproject.toml              # Phase 1: Project config
├── README.md                   
├── .env.example                
├── .gitignore                  
├── Dockerfile                  # Phase 3: Deployment
├── docker-compose.yml          
├── k8s/                        
│   └── deployment.yaml         
├── src/
│   └── specagent/
│       ├── __init__.py
│       ├── config.py           # Phase 1: Configuration
│       ├── nodes/              # Phase 2: LangGraph nodes
│       │   ├── __init__.py
│       │   ├── router.py
│       │   ├── retriever.py
│       │   ├── grader.py
│       │   ├── rewriter.py
│       │   ├── generator.py
│       │   └── hallucination.py
│       ├── graph/              # Phase 2: LangGraph workflow
│       │   ├── __init__.py
│       │   ├── state.py
│       │   └── workflow.py
│       ├── retrieval/          # Phase 1: Core retrieval
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── indexer.py
│       │   └── chunker.py
│       ├── evaluation/         # Phase 4: Eval harness
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── benchmark.py
│       ├── api/                # Phase 3: FastAPI
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── models.py
│       └── tracing/            # Phase 4: Observability
│           ├── __init__.py
│           └── phoenix.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
│   ├── ingest_data.py
│   ├── build_index.py
│   └── run_benchmark.py
├── data/
│   └── .gitkeep
└── docs/
    ├── architecture.md
    └── adr/                    # Architecture Decision Records
```

---

## 2. Task Decomposition Framework

### 2.1 The "One Module, One Session" Rule

Each Claude Code session should focus on **one module with one responsibility**. This keeps context focused and outputs reviewable.

| Task Size | Claude Code Suitability | Example |
|-----------|------------------------|---------|
| Single function | ✅ Excellent | "Write the `chunk_document` function" |
| Single module (~200 lines) | ✅ Good | "Implement `src/specagent/retrieval/chunker.py`" |
| Multiple related modules | ⚠️ Okay with care | "Implement the grader node and its tests" |
| Entire subsystem | ❌ Too broad | "Build the retrieval pipeline" |
| Architectural decisions | ❌ Not suitable | "Design the system architecture" |

### 2.2 Optimal Task Prompts by Component

#### Phase 1: Data Ingestion & Retrieval

```bash
# Task 1.1: Configuration module
claude "Create src/specagent/config.py with:
- Pydantic Settings class loading from environment variables
- Fields for: HF_API_KEY, EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2), 
  CHUNK_SIZE (default: 512), CHUNK_OVERLAP (default: 64), FAISS_INDEX_PATH, LLM_MODEL
- Validation for required fields
- Type hints throughout
Include a test file tests/unit/test_config.py with pytest"

# Task 1.2: Document chunker
claude "Create src/specagent/retrieval/chunker.py with:
- Function chunk_markdown(text: str, chunk_size: int, overlap: int) -> list[Chunk]
- Chunk dataclass with: content, metadata (source_file, section_header, chunk_index)
- Use langchain's RecursiveCharacterTextSplitter under the hood
- Preserve markdown section headers in metadata
- Include docstrings and type hints
Write tests in tests/unit/test_chunker.py covering: 
- Basic chunking, overlap behavior, metadata extraction, empty input handling"

# Task 1.3: Embedding client
claude "Create src/specagent/retrieval/embeddings.py with:
- Class HuggingFaceEmbedder that wraps the HF Inference API
- Method embed_texts(texts: list[str]) -> np.ndarray
- Batch processing with configurable batch_size (default 32)
- Retry logic with exponential backoff for rate limits (429 errors)
- Async version aembed_texts for concurrent processing
Use httpx for HTTP calls. Include type hints and docstrings.
Write tests using pytest-httpx to mock API responses."

# Task 1.4: FAISS indexer
claude "Create src/specagent/retrieval/indexer.py with:
- Class FAISSIndex with methods: build(chunks, embeddings), search(query_embedding, k), save(path), load(path)
- Store chunk metadata alongside vectors
- Use IndexFlatIP for cosine similarity (normalize vectors before indexing)
- Memory-efficient: use memory-mapped index for large datasets
Include tests that create a small index, add vectors, search, and verify results."
```

#### Phase 2: LangGraph Nodes

```bash
# Task 2.1: Graph state definition
claude "Create src/specagent/graph/state.py with:
- TypedDict class GraphState containing:
  - question: str
  - rewritten_question: Optional[str]
  - retrieved_chunks: list[RetrievedChunk]
  - graded_chunks: list[GradedChunk]  
  - generation: Optional[str]
  - citations: list[Citation]
  - route_decision: Literal['retrieve', 'reject']
  - rewrite_count: int (default 0)
  - hallucination_check: Literal['grounded', 'not_grounded', 'partial']
- Supporting dataclasses for RetrievedChunk, GradedChunk, Citation
Use Python 3.11+ typing. Include docstrings explaining each field's purpose."

# Task 2.2: Router node
claude "Create src/specagent/nodes/router.py with:
- Pydantic model RouteDecision with fields: route (Literal['retrieve', 'reject']), reasoning (str)
- Function router_node(state: GraphState) -> GraphState that:
  - Calls LLM with structured output to get RouteDecision
  - Uses the prompt from our PRD (I'll provide it)
  - Updates state['route_decision']
  - Returns updated state
- Use langchain_community.llms.HuggingFaceHub for LLM calls
Include tests with mocked LLM responses for both retrieve and reject cases."

# Task 2.3: Grader node  
claude "Create src/specagent/nodes/grader.py with:
- Pydantic model GradeResult with: relevant (Literal['yes', 'no']), confidence (float 0-1)
- Function grader_node(state: GraphState) -> GraphState that:
  - Iterates through state['retrieved_chunks']
  - Calls LLM to grade each chunk's relevance to the question
  - Populates state['graded_chunks'] with results
  - Calculates average confidence score
- Include the grading prompt from our PRD
Write tests covering: all relevant, all irrelevant, mixed results."

# Task 2.4: Rewriter node
claude "Create src/specagent/nodes/rewriter.py with:
- Function rewriter_node(state: GraphState) -> GraphState that:
  - Takes the original question and failed chunk summaries
  - Calls LLM to generate a rewritten, more specific question
  - Updates state['rewritten_question'] and increments state['rewrite_count']
  - Handles max rewrite limit (2)
Write tests for: successful rewrite, max rewrite limit reached."

# Task 2.5: Generator node
claude "Create src/specagent/nodes/generator.py with:
- Function generator_node(state: GraphState) -> GraphState that:
  - Takes graded chunks marked as relevant
  - Formats them into a context string with source citations
  - Calls LLM to generate answer with inline citations [TS XX.XXX §Y.Z]
  - Parses citations from response into state['citations']
  - Updates state['generation']
Include citation extraction regex and tests."

# Task 2.6: Hallucination checker
claude "Create src/specagent/nodes/hallucination.py with:
- Pydantic model HallucinationResult with: grounded (Literal['yes', 'no', 'partial']), 
  ungrounded_claims (list[str])
- Function hallucination_check_node(state: GraphState) -> GraphState that:
  - Compares generation against source chunks
  - Uses LLM-as-judge to verify factual grounding
  - Updates state['hallucination_check']
Write tests for grounded, not grounded, and partial cases."

# Task 2.7: Workflow assembly
claude "Create src/specagent/graph/workflow.py with:
- Function build_graph() -> CompiledGraph that:
  - Creates StateGraph with GraphState
  - Adds all nodes (router, retriever, grader, rewriter, generator, hallucination_check)
  - Adds conditional edges:
    - router: retrieve -> retriever, reject -> END
    - grader: low confidence -> rewriter, high confidence -> generator
    - rewriter: -> retriever (loop back)
    - hallucination_check: grounded -> END, not_grounded -> generator (retry)
  - Compiles and returns the graph
- Function run_query(question: str) -> dict with the full response
Include a visualization export using graph.get_graph().draw_mermaid()"
```

#### Phase 3: API & Deployment

```bash
# Task 3.1: FastAPI application
claude "Create src/specagent/api/main.py with:
- FastAPI app with /query POST endpoint
- Request model: QueryRequest(question: str, verbose: bool = False, max_rewrites: int = 2)
- Response model: QueryResponse(answer: str, citations: list, confidence: float, metadata: dict)
- Error handling for: off-topic queries, empty retrieval, hallucination detection
- /health GET endpoint for k8s liveness probe
- CORS middleware configured
Include OpenAPI documentation strings."

# Task 3.2: Dockerfile
claude "Create a multi-stage Dockerfile for specagent:
- Stage 1: Build with poetry, export requirements
- Stage 2: Runtime with python:3.11-slim
- Install only production dependencies
- Copy FAISS index from build context
- Run with uvicorn, 1 worker (memory constraint)
- Expose port 8000
- Health check using curl to /health
- Target final image size under 1GB"

# Task 3.3: Kubernetes manifests
claude "Create k8s/deployment.yaml with:
- Deployment with 1 replica (memory constraint)
- Resource limits: 4Gi memory, 2 CPU
- Liveness probe on /health
- Readiness probe on /health  
- Environment variables from ConfigMap and Secret
- Volume mount for FAISS index (PVC)
Create corresponding Service (ClusterIP) and ConfigMap."
```

#### Phase 4: Evaluation & Observability

```bash
# Task 4.1: RAGAS integration
claude "Create src/specagent/evaluation/metrics.py with:
- Function evaluate_e2e(test_dataset: list[dict]) -> dict that:
  - Runs RAGAS evaluation with Faithfulness, AnswerRelevancy, ContextPrecision
  - Returns metric scores as dictionary
- Function evaluate_retrieval(queries: list, ground_truth_docs: list) -> dict
  - Computes Recall@5, Recall@10, MRR
- Use ragas library, handle API rate limits
Include type hints and docstrings."

# Task 4.2: Benchmark runner
claude "Create scripts/run_benchmark.py with:
- Load TSpec-LLM benchmark questions from JSON
- Run each through the pipeline
- Compute accuracy by difficulty (easy/medium/hard)
- Generate markdown report with results table
- Save results to evaluation/results/{timestamp}.json
Use argparse for CLI: --dataset, --output-dir, --limit (for testing)"

# Task 4.3: Phoenix tracing setup
claude "Create src/specagent/tracing/phoenix.py with:
- Function setup_tracing() that initializes Phoenix with OpenTelemetry
- Decorator @traced that wraps functions with span creation
- Integration with LangChain callbacks for automatic LLM tracing
- Configuration for local Phoenix server endpoint
Reference the Phoenix + LangGraph integration docs."
```

---

## 3. Safety Guardrails

### 3.1 The Review-Before-Commit Rule

**Never commit Claude Code output directly.** Always:

1. **Read the generated code** — Understand what it does
2. **Run the tests** — Verify it works
3. **Check for security issues** — API keys, injection vulnerabilities
4. **Verify dependencies** — No unnecessary packages added
5. **Then commit** — With a meaningful message

```bash
# Workflow
claude "Implement the chunker module..."  # Claude generates code
git diff                                    # Review changes
pytest tests/unit/test_chunker.py          # Run tests
git add -p                                  # Selective staging
git commit -m "feat: add document chunker with metadata extraction"
```

### 3.2 Test-First Development Pattern

Ask Claude Code to write tests BEFORE implementation:

```bash
# Step 1: Generate tests first
claude "Write pytest tests for a FAISSIndex class that should:
- Build an index from chunks and embeddings
- Search by query embedding and return top-k results
- Save to and load from disk
- Handle edge cases: empty index, k > num_vectors
Put tests in tests/unit/test_indexer.py. 
Don't implement the class yet - just the tests."

# Step 2: Review tests, ensure they match requirements

# Step 3: Generate implementation
claude "Now implement src/specagent/retrieval/indexer.py 
to pass all the tests in tests/unit/test_indexer.py.
Here are the test contents: [paste tests]"

# Step 4: Run tests
pytest tests/unit/test_indexer.py -v
```

### 3.3 Isolation Pattern for Risky Operations

For operations that touch external services or data:

```bash
# Create isolated test environment
claude "Create a pytest fixture in tests/conftest.py that:
- Sets up a temporary directory for FAISS index
- Creates a mock HuggingFace API client using pytest-httpx
- Provides sample 3GPP document chunks for testing
- Cleans up after tests complete
Mark fixtures with appropriate scopes (function/module/session)"
```

### 3.4 Dependency Hygiene

Explicitly specify allowed dependencies:

```bash
# Good: Explicit about dependencies
claude "Implement the embeddings client using only these packages:
- httpx for HTTP requests
- numpy for array operations
- tenacity for retry logic
Do not add any other dependencies."

# Bad: Open-ended
claude "Implement the embeddings client"  # Might add unnecessary deps
```

---

## 4. Prompt Engineering for Claude Code

### 4.1 The Context Sandwich

Structure prompts with context → task → constraints:

```bash
claude "
CONTEXT:
I'm building an agentic RAG system for 3GPP telecom specifications.
The system uses LangGraph for orchestration and FAISS for vector search.
Here's the current project structure: [paste tree output]

TASK:
Create the retriever node that:
1. Takes a query from graph state
2. Embeds the query using our HuggingFaceEmbedder
3. Searches the FAISS index for top-10 similar chunks
4. Updates state with retrieved chunks

CONSTRAINTS:
- Use async/await for the embedding call
- Handle the case where index is not loaded
- Include type hints and docstrings
- Follow the node pattern from router.py (I'll paste it)

REFERENCE CODE:
[paste router.py as example of node pattern]
"
```

### 4.2 Iterative Refinement

Don't try to get it perfect in one shot:

```bash
# Round 1: Basic implementation
claude "Implement basic grader node that grades chunk relevance"

# Review output, identify issues

# Round 2: Add specific improvements  
claude "Update the grader node to:
- Batch LLM calls for efficiency (max 5 chunks per call)
- Add timeout handling (30s per call)
- Log grading decisions for debugging"

# Round 3: Add tests
claude "Add integration tests for the updated grader node 
that verify batching behavior and timeout handling"
```

### 4.3 Reference-Driven Generation

Provide examples of what you want:

```bash
claude "Create the generator node following this exact pattern:

EXAMPLE (router node):
\`\`\`python
def router_node(state: GraphState) -> GraphState:
    \"\"\"Route query to retrieval or rejection.\"\"\"
    llm = get_llm()
    result = llm.with_structured_output(RouteDecision).invoke(...)
    state['route_decision'] = result.route
    return state
\`\`\`

Now create generator_node with the same:
- Function signature pattern
- Docstring style
- State update pattern
- Error handling approach"
```

---

## 5. What NOT to Delegate to Claude Code

### 5.1 Architectural Decisions

❌ "Design the system architecture"
❌ "Choose between LangGraph and LlamaIndex"
❌ "Decide how to structure the evaluation pipeline"

These require your judgment and context about requirements, constraints, and tradeoffs.

### 5.2 Security-Sensitive Code

❌ "Implement authentication"
❌ "Handle API key management"
❌ "Set up secrets in Kubernetes"

Review security code extra carefully or write it yourself.

### 5.3 Complex Debugging

❌ "Fix this bug" (without clear reproduction steps)
❌ "Why isn't this working?"

Instead, narrow down the issue first, then ask for specific fixes:
✅ "The grader returns empty results when confidence < 0.5. Update the filtering logic to..."

### 5.4 Data Pipeline Operations

❌ "Download and process the TSpec-LLM dataset"
❌ "Build the FAISS index from the full corpus"

These involve large data transfers and long-running operations. Write scripts yourself and run them manually.

---

## 6. Recommended Development Sequence

### Week 1-2: Foundation

| Day | Task | Claude Code Usage |
|-----|------|-------------------|
| 1 | Project scaffold | ✅ Generate structure |
| 2 | Config & environment | ✅ Generate config.py |
| 3 | Chunker implementation | ✅ Generate + tests |
| 4 | Embeddings client | ✅ Generate + tests |
| 5 | FAISS indexer | ✅ Generate + tests |
| 6-7 | Data ingestion script | ⚠️ Outline only, implement manually |

### Week 3-4: Agentic Pipeline

| Day | Task | Claude Code Usage |
|-----|------|-------------------|
| 8 | Graph state definition | ✅ Generate state.py |
| 9 | Router node | ✅ Generate + tests |
| 10 | Grader node | ✅ Generate + tests |
| 11 | Rewriter node | ✅ Generate + tests |
| 12 | Generator node | ✅ Generate + tests |
| 13 | Hallucination checker | ✅ Generate + tests |
| 14 | Workflow assembly | ⚠️ Generate, review carefully |

### Week 5-6: API & Deployment

| Day | Task | Claude Code Usage |
|-----|------|-------------------|
| 15 | FastAPI endpoints | ✅ Generate main.py |
| 16 | Dockerfile | ✅ Generate |
| 17 | k8s manifests | ✅ Generate |
| 18 | Local testing | ❌ Manual |
| 19 | k3s deployment | ❌ Manual |
| 20 | Gradio UI | ✅ Generate basic UI |

### Week 7-8: Evaluation & Polish

| Day | Task | Claude Code Usage |
|-----|------|-------------------|
| 21 | RAGAS integration | ✅ Generate metrics.py |
| 22 | Benchmark runner | ✅ Generate script |
| 23 | Phoenix tracing | ✅ Generate setup |
| 24-25 | Run benchmarks | ❌ Manual |
| 26 | README & docs | ✅ Generate, then edit |
| 27-28 | Polish & demo | ❌ Manual |

---

## 7. Session Management Tips

### 7.1 Keep Sessions Focused

```bash
# Start a new session for each module
claude --new "Working on retrieval/chunker.py"

# Continue in same session for related follow-ups
claude "Add a function to extract section headers from markdown"

# New session for different module
claude --new "Working on nodes/router.py"
```

### 7.2 Provide Project Context

At the start of each session, give Claude Code the relevant context:

```bash
claude "
I'm continuing work on the specagent project.
Current file structure:
$(tree src/specagent -I __pycache__)

Dependencies installed:
$(cat pyproject.toml | grep -A 20 '\[tool.poetry.dependencies\]')

Now, implement..."
```

### 7.3 Save and Restore Context

For complex tasks spanning multiple sessions:

```bash
# Save context to a file
cat > .claude-context.md << 'EOF'
# SpecAgent Development Context

## Project Goal
Agentic RAG for 3GPP specifications

## Current State
- ✅ Retrieval pipeline complete
- 🔄 Working on LangGraph nodes
- ❌ API not started

## Key Files
- src/specagent/graph/state.py - Graph state definition
- src/specagent/nodes/router.py - Example node pattern

## Constraints
- 4GB RAM limit
- HuggingFace free tier API
- FAISS for vector store
EOF

# Reference in sessions
claude "Read .claude-context.md for project context, then implement the grader node"
```

---

## 8. Quality Checklist

Before considering a Claude Code-generated module complete:

- [ ] **Tests pass**: `pytest tests/unit/test_<module>.py -v`
- [ ] **Type hints**: `mypy src/specagent/<module>.py`
- [ ] **Linting**: `ruff check src/specagent/<module>.py`
- [ ] **Docstrings**: All public functions documented
- [ ] **No hardcoded values**: Config loaded from environment
- [ ] **Error handling**: Edge cases covered
- [ ] **Logging**: Appropriate log statements added
- [ ] **Dependencies**: Only approved packages used
- [ ] **Security**: No secrets in code, no unsafe operations

---

## 9. Example Full Session

Here's what a complete Claude Code session looks like:

```bash
# 1. Start session with context
$ claude --new

You: I'm building specagent, an agentic RAG for 3GPP specifications.
Here's my project structure:
$(tree src/specagent)

I need to implement the grader node. Here's the graph state it works with:
$(cat src/specagent/graph/state.py)

And here's the router node as a pattern to follow:
$(cat src/specagent/nodes/router.py)

Create src/specagent/nodes/grader.py that:
1. Takes retrieved chunks from state
2. Grades each for relevance using LLM with structured output
3. Uses this Pydantic model for grades:
   - relevant: Literal["yes", "no"]
   - confidence: float (0-1)
4. Updates state["graded_chunks"] with results
5. Follows the same patterns as router.py

Also create tests/unit/test_grader.py with mocked LLM responses.

Claude: [generates code]

# 2. Review generated code
$ cat src/specagent/nodes/grader.py
$ cat tests/unit/test_grader.py

# 3. Run tests
$ pytest tests/unit/test_grader.py -v

# 4. Fix any issues
You: The tests fail because the mock isn't returning the right structure.
Update the mock to return GradeResult objects instead of dicts.

Claude: [fixes code]

# 5. Verify and commit
$ pytest tests/unit/test_grader.py -v  # All pass
$ git add src/specagent/nodes/grader.py tests/unit/test_grader.py
$ git commit -m "feat(nodes): implement grader node with relevance scoring"
```

---

## 10. Efficiency Metrics

Track these to measure your Claude Code productivity:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Time per module | <2 hours | Timer per module |
| Test coverage | >80% | `pytest --cov` |
| First-attempt success rate | >70% | Tests pass without iteration |
| Lines of code reviewed | 100% | Self-discipline |
| Security issues found | 0 | Manual review + `bandit` |

---

## Summary

**The Golden Rules:**

1. **Decompose** — One module per session
2. **Test first** — Write tests before implementation
3. **Review always** — Never commit unreviewed code
4. **Iterate** — Refine in multiple rounds
5. **Isolate** — Mock external services
6. **Document** — Require docstrings and type hints

Claude Code accelerates development but doesn't replace engineering judgment. Use it as a force multiplier for well-defined tasks, not a replacement for architectural thinking.
