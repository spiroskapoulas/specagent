# Product Requirements Document: 3GPP SpecAgent

## Metadata

| Field | Value |
|-------|-------|
| **Product Name** | 3GPP SpecAgent |
| **Version** | 1.0.0 |
| **Author** | [Your Name] |
| **Date** | 2025-01-XX |
| **Status** | Draft |
| **Repository** | `github.com/[username]/3gpp-specagent` |

---

## 1. Executive Summary

**3GPP SpecAgent** is an agentic RAG (Retrieval-Augmented Generation) system that enables telecom engineers to query 3GPP Release 18 specifications using natural language. Unlike naive RAG implementations that achieve ~75% accuracy on telecom questions, SpecAgent employs an agentic architecture with question rewriting, document grading, and hallucination detection to target **85%+ accuracy** while maintaining sub-3-second response latency.

The system ingests the [TSpec-LLM dataset](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM)—a curated collection of 30,000+ 3GPP documents—and exposes a conversational interface for engineers working on 5G NR, LTE-Advanced, and network architecture implementations.

### Why This Matters

Telecom engineers spend **15-20 hours/week** navigating 3GPP specifications—documents notorious for dense cross-references, jargon-heavy prose, and interdependent standards. A single misinterpretation can result in non-compliant implementations costing millions in certification delays. SpecAgent reduces specification lookup time by 80% while providing traceable citations to authoritative sources.

---

## 2. Goals & Objectives

### 2.1 Business Goals

| Goal | Target | Measurement |
|------|--------|-------------|
| Reduce spec lookup time | 80% reduction | A/B testing with telecom engineers |
| Demonstrate portfolio-quality AI engineering | — | GitHub stars, interview callbacks |
| Showcase agentic RAG patterns | — | Architecture complexity, clean abstractions |

### 2.2 User Goals

| Goal | Description |
|------|-------------|
| **Instant answers** | Get accurate answers to 3GPP questions without reading 200-page specs |
| **Traceable citations** | Every answer links to specific spec sections for verification |
| **Handle ambiguity** | System reformulates vague questions into precise technical queries |

### 2.3 Success Metrics

#### Primary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Answer Accuracy** | ≥85% on TSpec-LLM benchmark | Baseline naive RAG achieves 71-75% (Nikbakht et al., 2024) |
| **End-to-End Latency (P95)** | <3 seconds | Competitive with manual search for single-fact queries |
| **Retrieval Recall@5** | ≥0.80 | Relevant chunk appears in top 5 results 80% of the time |

#### Secondary / Guardrail Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Hallucination Rate** | <5% | Measured via LLM-as-judge on 100 sampled responses |
| **Question Rewrite Trigger Rate** | 15-30% | Too low = not helping; too high = poor initial retrieval |
| **Infrastructure Cost** | <$0/month (free tier) | HuggingFace Inference API, no GPU hosting |
| **Memory Footprint** | <4GB RAM | k8s pod constraint |

---

## 3. Problem Statement & User Needs

### 3.1 Target Personas

#### Primary: Telecom Protocol Engineer

> **"I spend more time searching specs than writing code."**

- **Context**: Implements 5G NR features (MIMO, beamforming, RRC procedures)
- **Pain Points**:
  - Cross-references between TS 38.xxx documents are hard to follow
  - Searching PDFs for specific parameters yields irrelevant results
  - Version confusion between Release 15/16/17/18 features
- **Jobs-to-be-Done**:
  - *When I'm implementing a 5G feature, I want to find the exact spec reference so I can ensure compliance*
  - *When I encounter an unfamiliar acronym, I want its definition and context so I can understand the full picture*

#### Secondary: Network Architect

> **"I need to understand capability boundaries before designing solutions."**

- **Context**: Designs network architectures, evaluates vendor compliance
- **Pain Points**:
  - Specs don't clearly state what's mandatory vs. optional
  - Comparing features across releases is tedious
- **Jobs-to-be-Done**:
  - *When evaluating a vendor claim, I want to verify it against 3GPP specs so I can negotiate accurately*

### 3.2 Current Solutions & Gaps

| Solution | Limitation |
|----------|------------|
| **3GPP Portal Search** | Keyword-only, no semantic understanding, returns full documents |
| **PDF Ctrl+F** | No cross-reference traversal, misses synonyms |
| **ChatGPT/Claude (vanilla)** | Hallucinations on telecom specifics, no citations, knowledge cutoff |
| **Naive RAG** | 71-75% accuracy, no self-correction, poor on multi-hop questions |

---

## 4. Scope

### 4.1 In-Scope (MVP)

| Feature | Priority | Description |
|---------|----------|-------------|
| **Document Ingestion** | P0 | Ingest TSpec-LLM Rel-18 Markdown files, chunk, embed, index |
| **Agentic Query Pipeline** | P0 | Router → Retrieve → Grade → Rewrite (if needed) → Re-retrieve → Generate |
| **Hallucination Detection** | P0 | Post-generation check for factual grounding in retrieved chunks |
| **Citation Linking** | P0 | Every answer includes `[TS 38.XXX §Y.Z]` style references |
| **REST API** | P1 | FastAPI endpoint for programmatic access |
| **Simple Web UI** | P1 | Gradio/Streamlit chat interface for demos |
| **Docker Deployment** | P1 | Single-container deployment with docker-compose |
| **k8s Manifest** | P2 | Kubernetes deployment for k3s cluster |

### 4.2 Out-of-Scope (v1.0)

| Feature | Rationale | Future Version |
|---------|-----------|----------------|
| Multi-release support (Rel-15/16/17) | Complexity; start with Rel-18 only | v1.1 |
| Fine-tuned embedding model | Requires GPU training; use off-the-shelf first | v2.0 |
| Web search fallback | Focus on local corpus quality first | v1.2 |
| Multi-turn conversation memory | Single-turn first; add history later | v1.1 |
| PDF/image table extraction | TSpec-LLM already provides cleaned Markdown | v2.0 |
| User authentication | Demo/portfolio scope | v2.0 |

---

## 5. User Stories & Use Cases

### 5.1 Core User Stories (MoSCoW Prioritized)

#### Must Have

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| US-01 | As a protocol engineer, I want to ask "What is the maximum number of component carriers in NR?" so I can configure my implementation correctly | Returns answer with TS 38.XXX citation; accuracy verified against ground truth |
| US-02 | As a user, I want the system to reformulate my vague question "How does handover work?" into specific sub-queries so I get comprehensive results | System decomposes into "RRC handover procedure" + "Xn handover" + "NG handover"; retrieves relevant chunks for each |
| US-03 | As a user, I want to see which spec sections were used to generate the answer so I can verify correctness | Response includes clickable references to source documents |
| US-04 | As a user, I want the system to say "I don't know" rather than hallucinate when the answer isn't in the specs | Hallucination checker gates responses; <5% false positive rate |

#### Should Have

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| US-05 | As a developer, I want a REST API endpoint so I can integrate SpecAgent into my workflow | `POST /query` returns JSON with answer, citations, confidence score |
| US-06 | As a user, I want to see intermediate reasoning steps so I understand why the system retrieved certain documents | Optional `verbose=true` flag shows router decisions, grader scores |

#### Could Have

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| US-07 | As an architect, I want to compare features across releases so I can plan migration | Query: "What changed in PDCCH between Rel-17 and Rel-18?" returns diff-style answer |

### 5.2 Edge Cases & Failure Modes

| Scenario | Expected Behavior |
|----------|-------------------|
| **Query about non-3GPP topic** | Router detects off-topic; returns "This question is outside 3GPP specifications. I can help with telecom standards questions." |
| **Ambiguous acronym (e.g., "UE" vs "UE" in different contexts)** | Question rewriter adds context; retriever uses expanded query |
| **No relevant documents found** | Grader rejects all chunks; system returns "I couldn't find relevant information in Release 18 specifications." |
| **Contradictory information in specs** | Response acknowledges both sources with citations; flags potential spec ambiguity |
| **Very long/complex query** | Query decomposer splits into sub-queries; synthesizes final answer |

---

## 6. Functional Requirements

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              3GPP SpecAgent                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────────────────────────────────────────────┐   │
│  │   User   │───▶│                   FastAPI Gateway                     │   │
│  │  Query   │    │                    POST /query                        │   │
│  └──────────┘    └──────────────────────────┬───────────────────────────┘   │
│                                              │                               │
│                                              ▼                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        LangGraph Agentic Pipeline                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                                                                  │  │  │
│  │  │   ┌─────────┐     ┌───────────┐     ┌─────────┐                 │  │  │
│  │  │   │ ROUTER  │────▶│ RETRIEVER │────▶│ GRADER  │                 │  │  │
│  │  │   │         │     │           │     │         │                 │  │  │
│  │  │   │ Decide: │     │ FAISS     │     │ Score   │                 │  │  │
│  │  │   │ retrieve│     │ top-k=10  │     │ chunks  │                 │  │  │
│  │  │   │ or IDK  │     │           │     │ 0-1     │                 │  │  │
│  │  │   └─────────┘     └───────────┘     └────┬────┘                 │  │  │
│  │  │                                          │                       │  │  │
│  │  │                    ┌─────────────────────┼─────────────────────┐ │  │  │
│  │  │                    │                     ▼                     │ │  │  │
│  │  │                    │  ┌──────────────────────────────────────┐ │ │  │  │
│  │  │                    │  │         CONDITIONAL EDGE             │ │ │  │  │
│  │  │                    │  │  avg_score < 0.6 ──▶ REWRITER        │ │ │  │  │
│  │  │                    │  │  avg_score ≥ 0.6 ──▶ GENERATOR       │ │ │  │  │
│  │  │                    │  │  rewrite_count > 2 ──▶ GENERATOR     │ │ │  │  │
│  │  │                    │  └──────────────────────────────────────┘ │ │  │  │
│  │  │                    │                     │                     │ │  │  │
│  │  │         ┌──────────┴──────────┐          │                     │ │  │  │
│  │  │         ▼                     ▼          │                     │ │  │  │
│  │  │   ┌───────────┐         ┌───────────┐    │                     │ │  │  │
│  │  │   │ REWRITER  │         │ GENERATOR │◀───┘                     │ │  │  │
│  │  │   │           │         │           │                          │ │  │  │
│  │  │   │ LLM       │         │ LLM       │                          │ │  │  │
│  │  │   │ rewrite   │────────▶│ synthesize│                          │ │  │  │
│  │  │   │ query     │ (loop)  │ answer    │                          │ │  │  │
│  │  │   └───────────┘         └─────┬─────┘                          │ │  │  │
│  │  │                               │                                 │ │  │  │
│  │  │                               ▼                                 │ │  │  │
│  │  │                   ┌─────────────────────┐                       │ │  │  │
│  │  │                   │ HALLUCINATION CHECK │                       │ │  │  │
│  │  │                   │                     │                       │ │  │  │
│  │  │                   │ LLM verifies answer │                       │ │  │  │
│  │  │                   │ grounded in chunks  │                       │ │  │  │
│  │  │                   └──────────┬──────────┘                       │ │  │  │
│  │  │                              │                                  │ │  │  │
│  │  │              ┌───────────────┼───────────────┐                  │ │  │  │
│  │  │              ▼               ▼               ▼                  │ │  │  │
│  │  │         [grounded]    [not grounded]   [uncertain]              │ │  │  │
│  │  │              │               │               │                  │ │  │  │
│  │  │              ▼               ▼               ▼                  │ │  │  │
│  │  │          RESPOND      REGENERATE w/    ADD CAVEAT               │ │  │  │
│  │  │                       stricter prompt                           │ │  │  │
│  │  │                                                                 │ │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Data & Inference Layer                         │ │
│  │                                                                        │ │
│  │   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐ │ │
│  │   │   FAISS Index   │   │   HF Inference   │   │   TSpec-LLM Rel-18 │ │ │
│  │   │   (in-memory)   │   │   API (free)     │   │   Markdown Corpus  │ │ │
│  │   │                 │   │                  │   │                    │ │ │
│  │   │  ~500K vectors  │   │  Embeddings:     │   │   Series: 21-38    │ │ │
│  │   │  dim=384        │   │  all-MiniLM-L6   │   │   ~2000 docs       │ │ │
│  │   │                 │   │                  │   │                    │ │ │
│  │   │  <2GB RAM       │   │  LLM:            │   │                    │ │ │
│  │   │                 │   │  Mistral-7B      │   │                    │ │ │
│  │   └─────────────────┘   └──────────────────┘   └────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Specifications

#### 6.2.1 Document Ingestion Pipeline

**Input**: TSpec-LLM Rel-18 Markdown files from HuggingFace

**Processing Steps**:

1. **Download**: Clone dataset via `huggingface_hub` (gated access requires token)
2. **Parse**: Extract Markdown files from `3GPP-clean/Rel-18/` subdirectories
3. **Chunk**: Split documents using recursive character splitter
   - `chunk_size`: 512 tokens
   - `chunk_overlap`: 64 tokens
   - Preserve section headers in metadata
4. **Embed**: Generate embeddings via HuggingFace Inference API
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - Batch size: 32 (API rate limit consideration)
5. **Index**: Build FAISS index with `IndexFlatIP` (inner product for cosine similarity)
6. **Persist**: Save index + metadata to disk for container restart

**Output**: `faiss.index` (~1.5GB) + `metadata.json` (~200MB)

#### 6.2.2 Router Node

**Purpose**: Determine if query requires retrieval or should be rejected

**Implementation**:
```python
class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    route: Literal["retrieve", "reject"]
    reasoning: str
```

**Prompt Template**:
```
You are a router for a 3GPP specification assistant.
Determine if the following question relates to 3GPP/telecom standards.

Question: {question}

If the question is about 3GPP specifications, 5G, LTE, NR, RAN, core network,
or any telecom standards topic, route to "retrieve".
If the question is completely unrelated (e.g., cooking, sports), route to "reject".

Respond with JSON: {"route": "retrieve" | "reject", "reasoning": "..."}
```

#### 6.2.3 Retriever Node

**Purpose**: Fetch relevant document chunks

**Configuration**:
- `top_k`: 10 (retrieve 10 candidates for grading)
- `similarity_threshold`: 0.3 (minimum cosine similarity)

**Output Schema**:
```python
@dataclass
class RetrievedChunk:
    content: str
    spec_id: str          # e.g., "TS38.331"
    section: str          # e.g., "5.3.3"
    similarity_score: float
    chunk_id: str
```

#### 6.2.4 Grader Node

**Purpose**: Score relevance of each retrieved chunk to the query

**Implementation**:
```python
class GradeDocuments(BaseModel):
    """Binary relevance score with confidence."""
    relevant: Literal["yes", "no"]
    confidence: float = Field(ge=0.0, le=1.0)
```

**Prompt Template**:
```
You are a grader assessing relevance of a retrieved document chunk to a user question.

Question: {question}

Document chunk:
---
{document}
---

Does this document contain information relevant to answering the question?
Consider: exact matches, related concepts, prerequisite information.

Respond with JSON: {"relevant": "yes" | "no", "confidence": 0.0-1.0}
```

**Decision Logic**:
- Calculate average confidence of "yes" grades
- If `avg_confidence < 0.6` AND `rewrite_count < 2`: trigger rewriter
- If `avg_confidence >= 0.6` OR `rewrite_count >= 2`: proceed to generator

#### 6.2.5 Rewriter Node

**Purpose**: Reformulate query for better retrieval

**Prompt Template**:
```
You are a query rewriter for a 3GPP specification search system.

Original question: {question}

The retrieval system found these documents, but they may not be relevant:
{retrieved_chunks_summary}

Rewrite the question to be more specific and likely to match 3GPP specification language.
Consider:
- Expanding acronyms (e.g., "UE" → "User Equipment (UE)")
- Adding technical context (e.g., "handover" → "RRC connection handover procedure")
- Specifying the protocol layer or interface

Rewritten question:
```

**Constraints**:
- Maximum 2 rewrites per query (prevent infinite loops)
- Track rewrite history in state

#### 6.2.6 Generator Node

**Purpose**: Synthesize answer from graded chunks

**Prompt Template**:
```
You are a 3GPP specification expert. Answer the question using ONLY the provided context.

Question: {question}

Context (from 3GPP specifications):
---
{graded_chunks}
---

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using format: [TS XX.XXX §Y.Z]
3. If the context doesn't contain the answer, say "I don't have enough information"
4. Be precise and technical

Answer:
```

#### 6.2.7 Hallucination Checker Node

**Purpose**: Verify generated answer is grounded in retrieved chunks

**Implementation**:
```python
class HallucinationCheck(BaseModel):
    """Assess factual grounding of generated response."""
    grounded: Literal["yes", "no", "partial"]
    ungrounded_claims: list[str] = []
```

**Prompt Template**:
```
You are a fact-checker for a 3GPP specification assistant.

Source documents:
---
{retrieved_chunks}
---

Generated answer:
---
{generated_answer}
---

Verify if EVERY factual claim in the answer is supported by the source documents.

Respond with JSON:
{
  "grounded": "yes" | "no" | "partial",
  "ungrounded_claims": ["list of claims not found in sources"]
}
```

**Decision Logic**:
- `grounded == "yes"`: Return response with high confidence
- `grounded == "partial"`: Return response with caveat about unverified claims
- `grounded == "no"`: Regenerate with stricter prompt OR return "insufficient information"

### 6.3 API Specification

#### POST /query

**Request**:
```json
{
  "question": "What is the maximum number of component carriers in NR Rel-18?",
  "verbose": false,
  "max_rewrites": 2
}
```

**Response**:
```json
{
  "answer": "In NR Release 18, the maximum number of component carriers is 16 for downlink and 16 for uplink carrier aggregation. [TS 38.101-1 §5.5A]",
  "citations": [
    {
      "spec_id": "TS38.101-1",
      "section": "5.5A",
      "chunk_preview": "The UE shall support a maximum of 16 component carriers..."
    }
  ],
  "confidence": 0.92,
  "metadata": {
    "rewrites": 0,
    "chunks_retrieved": 10,
    "chunks_used": 3,
    "latency_ms": 2340
  }
}
```

**Error Response**:
```json
{
  "error": "off_topic",
  "message": "This question is outside 3GPP specifications. I can help with telecom standards questions."
}
```

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| P50 Latency | <2 seconds | End-to-end query time |
| P95 Latency | <3 seconds | End-to-end query time |
| P99 Latency | <5 seconds | Acceptable for complex rewrites |
| Throughput | 10 queries/minute | HuggingFace free tier rate limit |
| Cold Start | <30 seconds | FAISS index load time |

### 7.2 Scalability

| Aspect | Design Decision |
|--------|-----------------|
| Horizontal scaling | Stateless API; FAISS index read-only after build |
| Index size | ~500K vectors @ 384 dims = ~750MB RAM |
| Concurrent requests | Queue-based; single worker (free tier constraint) |

### 7.3 Reliability

| Requirement | Implementation |
|-------------|----------------|
| API retry logic | Exponential backoff for HF Inference API (429 errors) |
| Graceful degradation | Return cached responses if API unavailable |
| Health check | `/health` endpoint for k8s liveness probe |

### 7.4 Security

| Concern | Mitigation |
|---------|------------|
| API key exposure | Environment variables; never commit to repo |
| Prompt injection | Input sanitization; structured outputs only |
| Data privacy | No user data persisted; TSpec-LLM is public dataset |
| Rate limiting | 100 requests/hour per IP (prevent abuse) |

### 7.5 Observability

| Signal | Implementation |
|--------|----------------|
| Metrics | Prometheus: latency histograms, cache hit rate, rewrite frequency |
| Logs | Structured JSON; include trace_id per request |
| Traces | OpenTelemetry spans for each pipeline node |

---

## 8. Dependencies & Risks

### 8.1 External Dependencies

| Dependency | Risk Level | Mitigation |
|------------|------------|------------|
| HuggingFace Inference API | Medium | Cache embeddings; fallback to smaller local model |
| TSpec-LLM Dataset | Low | Dataset is versioned; pin to specific commit |
| LangGraph | Low | Well-maintained; pin version in requirements |
| FAISS | Low | Mature library; CPU-only version sufficient |

### 8.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HF free tier rate limits | High | Medium | Implement request queue; cache frequent queries |
| Memory exceeds 4GB | Medium | High | Profile during development; use memory-mapped FAISS if needed |
| Low accuracy on complex queries | Medium | High | Tune chunk size, add query decomposition node |
| LLM quality degradation | Low | Medium | Monitor hallucination rate; A/B test prompts |

### 8.3 Ethical Considerations

| Concern | Approach |
|---------|----------|
| Overreliance on AI | Clear disclaimers that answers should be verified against original specs |
| Bias in responses | TSpec-LLM is official 3GPP content; minimal bias risk |
| Misuse for non-compliant implementations | Out of scope; users are professionals with verification responsibility |

---

## 9. Timeline & Milestones

### Phase 1: Foundation (Week 1-2)

| Milestone | Deliverable | Exit Criteria |
|-----------|-------------|---------------|
| M1.1 | Data ingestion pipeline | Successfully chunks and embeds Rel-18 docs |
| M1.2 | FAISS index | Index builds in <10 minutes; loads in <30 seconds |
| M1.3 | Basic retrieval | Retrieves relevant chunks for sample queries |

### Phase 2: Agentic Pipeline (Week 3-4)

| Milestone | Deliverable | Exit Criteria |
|-----------|-------------|---------------|
| M2.1 | LangGraph workflow | Router → Retrieve → Grade → Generate working |
| M2.2 | Question rewriter | Triggers on low-confidence grades; improves recall |
| M2.3 | Hallucination checker | Catches >80% of obviously wrong answers |

### Phase 3: Productionization (Week 5-6)

| Milestone | Deliverable | Exit Criteria |
|-----------|-------------|---------------|
| M3.1 | FastAPI service | REST API passes integration tests |
| M3.2 | Docker image | Builds and runs locally; <4GB memory |
| M3.3 | k8s deployment | Deploys to k3s; health checks pass |
| M3.4 | Demo UI | Gradio app works; deployed to HuggingFace Spaces |

### Phase 4: Evaluation & Polish (Week 7-8)

| Milestone | Deliverable | Exit Criteria |
|-----------|-------------|---------------|
| M4.1 | Benchmark evaluation | Run TSpec-LLM question set; document accuracy |
| M4.2 | README & documentation | Professional README; architecture diagrams; ADRs |
| M4.3 | Portfolio presentation | Demo video; "What I Learned" writeup |

---

## 10. Future Enhancements Roadmap

This section outlines strategic enhancements beyond the MVP, prioritized by impact and complexity. These demonstrate forward-thinking architecture decisions and provide natural "Phase 2" talking points for interviews.

### 10.1 Phase 2: Prompt Optimization with DSPy

**Target Timeline**: v1.1 (Week 9-12)

**Problem Statement**

The MVP uses hand-crafted prompts for all nodes (router, grader, rewriter, generator, hallucination checker). While functional, these prompts are:
- Brittle to model changes (switching from Mistral to Llama requires re-tuning)
- Suboptimal without systematic experimentation
- Time-consuming to iterate manually

**Proposed Solution: DSPy Integration**

[DSPy](https://github.com/stanfordnlp/dspy) (Declarative Self-improving Python) replaces manual prompt engineering with programmatic optimization. Instead of writing prompts, you define **Signatures** and let DSPy's compiler find optimal prompts through metric-driven iteration.

**Architecture: Hybrid LangGraph + DSPy**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Phase 2: LangGraph + DSPy Hybrid                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      LangGraph (Orchestration)                       │   │
│   │                                                                      │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │   │
│   │   │  Router  │───▶│Retriever │───▶│  Grader  │───▶│ Rewriter │      │   │
│   │   │   Node   │    │   Node   │    │   Node   │    │   Node   │      │   │
│   │   └────┬─────┘    └──────────┘    └────┬─────┘    └────┬─────┘      │   │
│   │        │                               │               │             │   │
│   │        ▼                               ▼               ▼             │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │                  DSPy Modules (Optimized LLM Calls)          │   │   │
│   │   │                                                              │   │   │
│   │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │   │
│   │   │  │ RouterSig   │  │ GraderSig   │  │ RewriterSig         │  │   │   │
│   │   │  │             │  │             │  │                     │  │   │   │
│   │   │  │ question ──▶│  │ question ──▶│  │ question ──────────▶│  │   │   │
│   │   │  │ route       │  │ document ──▶│  │ failed_chunks ─────▶│  │   │   │
│   │   │  │             │  │ relevant    │  │ rewritten_question  │  │   │   │
│   │   │  │             │  │ confidence  │  │                     │  │   │   │
│   │   │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │   │
│   │   │                                                              │   │   │
│   │   │         Prompts automatically optimized by DSPy              │   │   │
│   │   │         using TSpec-LLM benchmark as training signal         │   │   │
│   │   └──────────────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Approach**

Step 1: Define DSPy Signatures for each node

```python
import dspy
from typing import Literal

class RouteQuery(dspy.Signature):
    """Determine if a question relates to 3GPP/telecom specifications."""
    question: str = dspy.InputField(desc="User's natural language question")
    route: Literal["retrieve", "reject"] = dspy.OutputField(
        desc="'retrieve' for 3GPP questions, 'reject' for off-topic"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of routing decision")

class GradeDocument(dspy.Signature):
    """Assess if a document chunk is relevant to answering a question."""
    question: str = dspy.InputField()
    document: str = dspy.InputField(desc="Retrieved chunk from 3GPP specification")
    relevant: Literal["yes", "no"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")

class RewriteQuery(dspy.Signature):
    """Reformulate a question to improve retrieval from 3GPP specifications."""
    original_question: str = dspy.InputField()
    failed_context: str = dspy.InputField(desc="Summary of irrelevant chunks retrieved")
    rewritten_question: str = dspy.OutputField(
        desc="More specific question using 3GPP terminology"
    )
```

Step 2: Create DSPy Modules with Chain-of-Thought reasoning

```python
class OptimizedGrader(dspy.Module):
    def __init__(self):
        self.grade = dspy.ChainOfThought(GradeDocument)
    
    def forward(self, question: str, document: str) -> dspy.Prediction:
        return self.grade(question=question, document=document)

class OptimizedRouter(dspy.Module):
    def __init__(self):
        self.route = dspy.Predict(RouteQuery)
    
    def forward(self, question: str) -> dspy.Prediction:
        return self.route(question=question)
```

Step 3: Optimize using TSpec-LLM benchmark

```python
from dspy.teleprompt import BootstrapFewShot, MIPROv2

# Define metric for optimization
def grader_metric(example, prediction, trace=None):
    """Metric: Does grader agree with human label?"""
    return prediction.relevant == example.relevant

# Load training data from TSpec-LLM benchmark
trainset = load_tspec_grader_examples()  # 100 labeled (question, chunk, relevant) triples

# Optimize grader prompts
teleprompter = BootstrapFewShot(metric=grader_metric, max_bootstrapped_demos=4)
optimized_grader = teleprompter.compile(OptimizedGrader(), trainset=trainset)

# Save optimized module
optimized_grader.save("models/optimized_grader.json")
```

Step 4: Integrate optimized modules into LangGraph nodes

```python
# Load pre-optimized DSPy module
optimized_grader = OptimizedGrader()
optimized_grader.load("models/optimized_grader.json")

def grader_node(state: GraphState) -> GraphState:
    """LangGraph node using DSPy-optimized grading."""
    graded_chunks = []
    for chunk in state["retrieved_chunks"]:
        result = optimized_grader(
            question=state["question"],
            document=chunk.content
        )
        graded_chunks.append({
            "chunk": chunk,
            "relevant": result.relevant,
            "confidence": result.confidence
        })
    state["graded_chunks"] = graded_chunks
    return state
```

**Expected Benefits**

| Metric | Before (Manual Prompts) | After (DSPy Optimized) | Improvement |
|--------|------------------------|------------------------|-------------|
| Grader F1 | ~0.85 | ~0.92 | +8% |
| Router Accuracy | ~0.95 | ~0.98 | +3% |
| Rewrite Improvement Rate | ~60% | ~75% | +25% |
| Prompt Iteration Time | Hours (manual) | Minutes (automated) | 10x faster |

**Considerations & Constraints**

| Consideration | Mitigation |
|---------------|------------|
| **API Cost**: DSPy optimization requires many LLM calls | Run optimization offline with cached responses; use smaller model for optimization, deploy with larger |
| **Rate Limits**: HuggingFace free tier limits | Batch optimization runs during off-peak; cache intermediate results |
| **Debugging Complexity**: Two abstraction layers | Log both LangGraph state and DSPy traces to Phoenix |
| **Cold Start**: Optimized prompts must be pre-computed | Save compiled modules to disk; load at container startup |

**Success Criteria for Phase 2**

- [ ] All 5 nodes converted to DSPy Signatures
- [ ] Optimization pipeline runs end-to-end
- [ ] ≥5% accuracy improvement over manual prompts on TSpec-LLM benchmark
- [ ] Optimization artifacts versioned and reproducible
- [ ] Documentation of prompt evolution (before/after examples)

---

### 10.2 Phase 2: Multi-Release Support

**Target Timeline**: v1.1 (Week 10-12)

**Problem**: MVP only supports Release 18. Telecom engineers often need to compare features across releases or query older specifications.

**Proposed Solution**:
- Ingest Rel-15, Rel-16, Rel-17 from TSpec-LLM
- Add release filter to retrieval: `"What is X in Release 17?"`
- Support cross-release queries: `"What changed in PDCCH between Rel-17 and Rel-18?"`

**Architecture Changes**:
- Chunk metadata includes `release_version` field
- Router detects release-specific queries
- New `release_comparator` node for diff-style answers

---

### 10.3 Phase 2: Query Decomposition

**Target Timeline**: v1.2 (Week 12-14)

**Problem**: Complex multi-hop questions fail because they require information from multiple spec sections.

**Example**: *"What are the security requirements for the handover procedure in NR standalone mode?"*

This requires:
1. Finding handover procedure (TS 38.300)
2. Finding security requirements for handover (TS 33.501)
3. Synthesizing both

**Proposed Solution**: Add `query_decomposer` node before router

```python
class DecomposeQuery(dspy.Signature):
    """Break down complex questions into simpler sub-queries."""
    complex_question: str = dspy.InputField()
    sub_queries: list[str] = dspy.OutputField(
        desc="List of 1-4 simpler questions that together answer the complex question"
    )
    synthesis_strategy: str = dspy.OutputField(
        desc="How to combine sub-query answers into final response"
    )
```

---

### 10.4 Phase 3: Fine-Tuned Embedding Model

**Target Timeline**: v2.0 (Month 3+)

**Problem**: Generic embedding models (`all-MiniLM-L6-v2`) don't understand telecom terminology well. "UE" (User Equipment), "gNB" (gNodeB), and "NR" (New Radio) may not embed close to their conceptual meanings.

**Proposed Solution**:
- Fine-tune embedding model on 3GPP corpus using contrastive learning
- Use TSpec-LLM as training data
- Target: Sentence-BERT style fine-tuning

**Expected Impact**: 10-15% improvement in Recall@10

**Constraint**: Requires GPU for training (not available on current k3s cluster)

---

### 10.5 Phase 3: Multimodal Support

**Target Timeline**: v2.0 (Month 3+)

**Problem**: 3GPP specs contain diagrams, flowcharts, and tables that are lost in text-only RAG.

**Proposed Solution**:
- Extract images from original DOCX files in TSpec-LLM
- Use vision-language model (e.g., LLaVA, GPT-4V) to generate text descriptions
- Index both text and image descriptions
- For queries about procedures/flows, retrieve relevant diagrams

**Example**: *"Show me the RRC connection establishment flow"* → Returns flowchart image with explanation

---

### 10.6 Enhancement Priority Matrix

| Enhancement | Impact | Effort | Dependencies | Priority |
|-------------|--------|--------|--------------|----------|
| DSPy Prompt Optimization | High | Medium | Baseline accuracy established | P1 |
| Multi-Release Support | High | Medium | Additional storage (~5GB) | P1 |
| Query Decomposition | Medium | Medium | DSPy integration | P2 |
| Fine-Tuned Embeddings | High | High | GPU access | P3 |
| Multimodal Support | Medium | High | Vision model API | P3 |

---

### 10.7 Interview Talking Points for Future Enhancements

**On DSPy**:
> "For the MVP, I hand-tuned prompts because it let me iterate quickly and understand the failure modes. In Phase 2, I'd integrate DSPy to automatically optimize prompts against my evaluation benchmark. This separates the 'what should the LLM do' (Signatures) from 'how to ask it' (optimized prompts), making the system more maintainable and model-agnostic."

**On Not Over-Engineering**:
> "I deliberately scoped the MVP to prove the agentic architecture works before adding optimization layers. DSPy, fine-tuned embeddings, and multimodal support are all valuable, but they're enhancements—not prerequisites for a working system."

**On Technical Depth**:
> "The DSPy integration isn't just 'plug and play.' You need to design metrics that align with your actual success criteria, curate training examples that cover edge cases, and handle the compilation cost. I'd run optimization offline and version the compiled modules like model artifacts."

---

---

## 11. Appendices

### 11.1 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Framework | LangGraph | Native support for agentic workflows; conditional edges |
| Vector Store | FAISS (CPU) | Lightweight; no external service; fits in 4GB |
| Embeddings | `all-MiniLM-L6-v2` | Good quality; fast; 384 dims (memory efficient) |
| LLM | Mistral-7B-Instruct (HF API) | Best quality on free tier; good at structured output |
| API | FastAPI | Async support; automatic OpenAPI docs |
| UI | Gradio | Quick prototyping; native HuggingFace Spaces support |
| Containerization | Docker | Standard; easy k8s integration |
| Orchestration | k3s | Lightweight k8s for single-node deployments |
| Observability | Arize Phoenix | Open-source tracing; OpenTelemetry native |
| Evaluation | RAGAS + agentevals | Industry-standard RAG metrics + trajectory evaluation |
| *Phase 2* | DSPy | Prompt optimization; declarative LLM programming |

### 11.2 3GPP Series Reference

| Series | Domain | Rel-18 Document Count (est.) |
|--------|--------|------------------------------|
| 21 | Requirements | ~20 |
| 22 | Service aspects | ~50 |
| 23 | Technical realization | ~100 |
| 24 | Signalling protocols (CT) | ~80 |
| 26 | Codecs | ~40 |
| 28 | Signalling protocols (evolved) | ~60 |
| 29 | Core network protocols | ~120 |
| 32 | Charging & OAM | ~80 |
| 33 | Security | ~60 |
| 36 | LTE radio | ~150 |
| 37 | Radio (multiple RATs) | ~50 |
| 38 | 5G NR | ~200 |

### 11.3 Sample Evaluation Questions

From TSpec-LLM benchmark (Nikbakht et al., 2024):

1. **Easy**: "What does PDCCH stand for?" → Physical Downlink Control Channel
2. **Medium**: "What is the maximum number of HARQ processes in NR?" → 16 (for FDD)
3. **Hard**: "Explain the RRC connection re-establishment procedure when T311 expires"

### 11.4 Open Questions

| Question | Owner | Status |
|----------|-------|--------|
| Should we support Rel-17 alongside Rel-18? | Product | Deferred to v1.1 |
| How to handle spec updates (TSpec-LLM refreshes)? | Eng | Need versioning strategy |
| Do we need query decomposition for multi-hop questions? | Eng | Test with benchmark first |
| What's the right chunk size for telecom specs? | Eng | Experiment: 256/512/1024 |

### 11.5 References

1. Nikbakht, R., Benzaghta, M., & Geraci, G. (2024). *TSpec-LLM: An Open-source Dataset for LLM Understanding of 3GPP Specifications*. arXiv:2406.01768
2. NVIDIA. (2024). *Build an Agentic RAG Pipeline with Llama 3.1 and NVIDIA NeMo Retriever NIMs*. NVIDIA Developer Blog.
3. Khattab, O., et al. (2024). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. ICLR 2024.
4. LangGraph Documentation: https://langchain-ai.github.io/langgraph/
5. DSPy Documentation: https://dspy.ai/
6. Arize Phoenix Documentation: https://docs.arize.com/phoenix
7. RAGAS Documentation: https://docs.ragas.io

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-01-XX | [Your Name] | Initial draft |
